import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import wrappers

import torch
from torch import nn
from torch import distributions as torchd
to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):

  def __init__(self, config, logger):
    super(Dreamer, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = count_steps(config.traindir)
    self._k = config.ensemble_number
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    # self._wm = models.WorldModel(self._step, config)
    self._ensemble_wm = models.EnsembleWorldModel(self._k, self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._ensemble_wm, config.behavior_stop_grad)
    reward = lambda f, s, a: np.array([wmi.heads['reward'](f).mean 
                                       for wmi in self._ensemble_wm._wms]).mean()
    # reward = lambda f, s, a: self._wm.heads['reward'](f).mean
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._ensemble_wm, reward),
    )[config.expl_behavior]() # greedy

  def __call__(self, obs, reset, state=None, reward=None, training=True):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        # Different World Model vedios should be logged respectively
        openl = self._ensemble_wm._wms[0].video_pred(next(self._dataset))
        self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

    policy_output, state = self._policy(obs, state, training)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return policy_output, state

  def _policy(self, obs, state, training):
    if state is None:
      batch_size = len(obs['image'])
      latent = [wmi.dynamics.initial(len(obs['image'])) for wmi in self._ensemble_wm._wms]
      action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
    else:
      latent, action = state
    embed = [wmi.encoder(wmi.preprocess(obs)) for wmi in self._ensemble_wm._wms]
    latent = [self._ensemble_wm._wms[i].dynamics.obs_step(
        latent[i], action, embed[i], self._config.collect_dyn_sample)[0] for i in range(self._k)]
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat_merge = torch.cat([self._ensemble_wm._wms[i].dynamics.get_feat(latent[i]) for i in range(self._k)], dim=-1)
    if not training:
      actor = self._task_behavior.actor(feat_merge)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat_merge)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat_merge)
      action = actor.sample()
    logprob = actor.log_prob(action)
    latent = [{k: v.detach()  for k, v in d.items()} for d in latent]
    action = action.detach()
    if self._config.actor_dist == 'onehot_gumble':
      action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  def _train(self, data, step):
    metrics = {}
    if step < 50000:
      post, context, mets = self._ensemble_wm._train(data)
      metrics.update(mets)
    else:
      # post = self._wm._train(data, train=False)
      # start = post
      post, context, mets = self._ensemble_wm._train(data)
      metrics.update(mets)
      start = post
      def reward(f, s, a):
        rewards = [self._ensemble_wm._wms[i].heads['reward'](
          self._ensemble_wm._wms[i].dynamics.get_feat(s[i])).mode() for i in range(self._k)]
        return torch.stack(rewards, dim=0).mean(dim=0)
      # reward = lambda f, s, a: np.array([self._ensemble_wm._wms[i].heads['reward'](
      #     self._ensemble_wm._wms[i].dynamics.get_feat(s[i])).mode() for i in range(self._k)]).mean()
      metrics.update(self._task_behavior._train(start, reward)[-1])
    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)
    if step % 1000 == 0:
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        # openl = self._wm.video_pred(data)
        # self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

  def make_kfolddataset(self, episodes):
    # import ipdb; ipdb.set_trace()
    keys = episodes.keys()
    self._part_keys = self.split_dict_into_k_parts(list(keys), self._k)
    generator = self.sample_kepisodes(episodes, self._config.batch_length, self._config.oversample_ends)
    dataset = self.from_kfoldgenerator(generator, self._config.batch_size)
    self._dataset = dataset

  def from_kfoldgenerator(self, generator, batch_size):
    while True:
      batches = []
      for _ in range(batch_size):
        data = next(generator)
        batches.append(next(generator))
      batches = [[listi[i] for listi in batches] for i in range(self._k)]
      datas = []
      for batch in batches:
        data = {}
        for key in batch[0].keys():
          data[key] = []
          for i in range(batch_size):
            data[key].append(batch[i][key])
          data[key] = np.stack(data[key], 0)
        datas.append(data)
      yield datas

  def sample_kepisodes(self, episodes, length=None, balance=False, seed=0):
    random = np.random.RandomState(seed)
    while True:
      data = []
      for i in range(self._k):
        episode_key = random.choice(list(episodes.keys()))
        while episode_key in self._part_keys[i]:
          episode_key = random.choice(list(episodes.keys()))
        episode = episodes[episode_key]
        if length:
          total = len(next(iter(episode.values())))
          available = total - length
          if available < 1:
            print(f'Skipped short episode of length {available}.')
            continue
          if balance:
            index = min(random.randint(0, total), available)
          else:
            index = int(random.randint(0, available + 1))
          episode = {k: v[index: index + length] for k, v in episode.items()}
          data.append(episode)
      yield data

  def split_dict_into_k_parts(self, keys, k):
    import random
    random.shuffle(keys)
    n = len(keys)
    part_size = n // k
    parts = []
    for i in range(k):
        start_index = i * part_size
        end_index = start_index + part_size if i < k - 1 else n
        part_keys = keys[start_index:end_index]
        parts.append(part_keys)
    return parts


def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
  generator = tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, config.size,
        grayscale=config.grayscale,
        life_done=False and ('train' in mode),
        sticky_actions=True,
        all_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'dmlab':
    env = wrappers.DeepMindLabyrinth(
        task,
        mode if 'train' in mode else 'test',
        config.action_repeat)
    env = wrappers.OneHotAction(env)
  elif suite == "metaworld":
      task = "-".join(task.split("_"))
      env = wrappers.MetaWorld(
          task,
          config.seed,
          config.action_repeat,
          config.size,
          config.camera,
      )
      env = wrappers.NormalizeActions(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  if (mode == 'train') or (mode == 'eval'):
    callbacks = [functools.partial(
        process_episode, config, logger, mode, train_eps, eval_eps)]
    env = wrappers.CollectDataset(env, callbacks, logger=logger, mode=mode)
  env = wrappers.RewardObs(env)
  return env


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


def main(config):
  logdir = pathlib.Path(config.logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.act = getattr(torch.nn, config.act)

  print('Logdir', logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)
  step = count_steps(config.traindir)
  step = 0
  logger = tools.Logger(logdir, step)

  print('Create envs.')
  if config.offline_traindir:
    directory = config.offline_traindir.format(**vars(config))
  else:
    directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)
  if config.offline_evaldir:
    directory = config.offline_evaldir.format(**vars(config))
  else:
    directory = config.evaldir
  eval_eps = tools.load_episodes(directory, limit=1)
  make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
  train_envs = [make('train') for _ in range(config.envs)]
  eval_envs = [make('eval') for _ in range(config.envs)]
  acts = train_envs[0].action_space
  config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  # if not config.offline_traindir:
  #   prefill = max(0, config.prefill - count_steps(config.traindir))
  #   print(f'Prefill dataset ({prefill} steps).')
  #   if hasattr(acts, 'discrete'):
  #     random_actor = tools.OneHotDist(torch.zeros_like(torch.Tensor(acts.low))[None])
  #   else:
  #     random_actor = torchd.independent.Independent(
  #         torchd.uniform.Uniform(torch.Tensor(acts.low)[None],
  #                                torch.Tensor(acts.high)[None]), 1)
  #   def random_agent(o, d, s, r):
  #     action = random_actor.sample()
  #     logprob = random_actor.log_prob(action)
  #     return {'action': action, 'logprob': logprob}, None
  #   # tools.simulate(random_agent, train_envs, prefill)
  #   tools.simulate(random_agent, eval_envs, episodes=config.eval_num)
  #   # logger.step = config.action_repeat * count_steps(config.traindir)

  print('Simulate agent.')
  train_dataset = make_dataset(train_eps, config)
  eval_dataset = make_dataset(eval_eps, config)

  # button_press  coffee_button  door_close  drawer_open  faucet_open  handle_press  handle_pull  plate_slide  reach_wall  window_close
  # ['coffee_button', 'door_close', 'faucet_open', 'handle_press', 'reach_wall', 'window_close']
  # button_press   drawer_open   handle_pull   plate_slide

  target_root = pathlib.Path(config.target_root).expanduser()
  # target_root = pathlib.Path('/data/mtpan/dataset/datasets_10_dreamerv3_100eps/plate_slide/eval_eps').expanduser()
  target_eps = tools.load_episodes(target_root, limit=config.dataset_size)
  # import ipdb; ipdb.set_trace()
  # target_dataset = make_dataset(target_eps, config)

  agent = Dreamer(config, logger).to(config.device)
  agent.make_kfolddataset(target_eps)
  agent.requires_grad_(requires_grad=False)
  if (logdir / 'latest_model.pt').exists():
    agent.load_state_dict(torch.load(logdir / 'latest_model.pt'))
    agent._should_pretrain._once = False
    print("Loaded Latest Model!")

  state = None
  for i in range(500000):
    if i >= 50000 and i % 3000 == 0:
      logger.write()
      print('Start evaluation.')
      eval_policy = functools.partial(agent, training=False)
      tools.simulate(eval_policy, eval_envs, episodes=config.eval_num)
      for j in range(config.ensemble_number):
        video_pred = agent._ensemble_wm._wms[j].video_pred(next(eval_dataset))
        logger.video(f'eval_openl{j}', to_np(video_pred))
    print('Start training.')
    print(i)
    agent._train(next(agent._dataset), i)
    # if i % 5000 == 0:
    #   torch.save(agent.state_dict(), logdir / 'latest_model.pt')
    if i % 100000 == 0 and i != 0:
      torch.save(agent.state_dict(), logdir / f'latest_model_{i}.pt')
    torch.save(agent.state_dict(), logdir / 'latest_model.pt')
    logger.step += 1
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(parser.parse_args(remaining))
