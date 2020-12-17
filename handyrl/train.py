# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Paper that proposed VTrace algorithm
# https://arxiv.org/abs/1802.01561
# Official code
# https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py

# training

import os
import time
import copy
import threading
import random
import bz2
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

from .environment import prepare_env, make_env
from .util import map_r, bimap_r, trimap_r, rotate, type_r
from .model import to_torch, to_gpu_or_not, RandomModel
from .model import DuelingNet as Model
from .connection import MultiProcessWorkers, MultiThreadWorkers
from .connection import accept_socket_connections
from .worker import Workers


def make_batch(episodes, args):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes
        args (dict): training configuration

    Returns:
        dict: PyTorch input and target tensors

    Note:
        Basic data shape is (T, B, P, ...) .
        (T is time length, B is batch size, P is player count)
    """

    obss, datum = [], []

    for ep in episodes:
        # target player and turn index
        moments_ = sum([pickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
        moments = moments_[ep['start'] - ep['base']:ep['end'] - ep['base']]
        players = sorted([player for player in moments[0]['observation'].keys() if player >= 0])

        obs_zeros = map_r(moments[0]['observation'][moments[0]['turn']], lambda o: np.zeros_like(o))  # template for padding
        if args['observation']:
            # replace None with zeros
            obs = [[(lambda x : (x if x is not None else obs_zeros))(m['observation'][pl]) for pl in players] for m in moments]
        else:
            obs = [[m['observation'][m['turn']]] for m in moments]
        obs = rotate(obs)  # (T, P, ..., ...) -> (P, ..., T, ...)
        obs = rotate(obs)  # (T, ..., P, ...) -> (..., P, T, ...)
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))

        # datum that is not changed by training configuration
        v = np.array(
            [[m['value'][player] or 0 for player in players] for m in moments],
            dtype=np.float32
        ).reshape(-1, len(players))
        tmsk = np.eye(len(players))[[m['turn'] for m in moments]]
        pmsk = np.array([m['pmask'] for m in moments])
        vmsk = np.ones_like(tmsk) if args['observation'] else tmsk

        act = np.array([m['action'] for m in moments]).reshape(-1, 1)
        p = np.array([m['policy'] for m in moments])
        progress = np.arange(ep['start'], ep['end'], dtype=np.float32) / ep['total']

        traj_steps = len(tmsk)
        ret = np.array(ep['reward'], dtype=np.float32).reshape(1, -1)

        # pad each array if step length is short
        if traj_steps < args['forward_steps']:
            pad_len = args['forward_steps'] - traj_steps
            obs = map_r(obs, lambda o: np.pad(o, [(0, pad_len)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0))
            v = np.concatenate([v, np.tile(ret, [pad_len, 1])])
            tmsk = np.pad(tmsk, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            pmsk = np.pad(pmsk, [(0, pad_len), (0, 0)], 'constant', constant_values=1e32)
            vmsk = np.pad(vmsk, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            act = np.pad(act, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            p = np.pad(p, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            progress = np.pad(progress, [(0, pad_len)], 'constant', constant_values=1)

        obss.append(obs)
        datum.append((tmsk, pmsk, vmsk, act, p, v, ret, progress))

    tmsk, pmsk, vmsk, act, p, v, ret, progress = zip(*datum)

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    tmsk = to_torch(np.array(tmsk))
    pmsk = to_torch(np.array(pmsk))
    vmsk = to_torch(np.array(vmsk))
    act = to_torch(np.array(act))
    p = to_torch(np.array(p))
    v = to_torch(np.array(v))
    ret = to_torch(np.array(ret))
    progress = to_torch(np.array(progress))

    return {
        'observation': obs, 'tmask': tmsk, 'pmask': pmsk, 'vmask': vmsk,
        'action': act, 'policy': p, 'value': v, 'return': ret, 'progress': progress,
    }


def forward_prediction(model, hidden, batch, obs_mode):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: calculated policy and value
    """

    observations = batch['observation']  # (B, T, P, ...)

    if hidden is None:
        # feed-forward neural network
        obs = map_r(observations, lambda o: o.view(-1, *o.size()[3:]))
        t_policies, t_values, _ = model(obs, None)
    else:
        # sequential computation with RNN
        bmasks = torch.clamp(batch['tmask'] + batch['vmask'], 0, 1)  # (B, T, P)

        t_policies, t_values = [], []
        for t in range(batch['tmask'].size(1)):
            obs = map_r(observations, lambda o: o[:, t].reshape(-1, *o.size()[3:]))  # (..., B * P, ...)
            bmask_ = bmasks[:, t]
            bmask = map_r(hidden, lambda h: bmask_.view(*h.size()[:2], *([1] * (len(h.size()) - 2))))
            hidden_ = bimap_r(hidden, bmask, lambda h, m: h * m)  # (..., B, P, ...)
            if obs_mode:
                hidden_ = map_r(hidden_, lambda h: h.view(-1, *h.size()[2:]))  # (..., B * P, ...)
            else:
                hidden_ = map_r(hidden_, lambda h: h.sum(1))  # (..., B * 1, ...)
            t_policy, t_value, next_hidden = model(obs, hidden_)
            t_policies.append(t_policy)
            t_values.append(t_value)
            next_hidden = bimap_r(next_hidden, hidden, lambda nh, h: nh.view(h.size(0), -1, *h.size()[2:]))  # (..., B, P or 1, ...)
            hidden = trimap_r(hidden, next_hidden, bmask, lambda h, nh, m: h * (1 - m) + nh * m)
        t_policies = torch.stack(t_policies, dim=1)
        t_values = torch.stack(t_values, dim=1)

    # gather turn player's policies
    t_policies = t_policies.view(*batch['tmask'].size()[:2], -1, t_policies.size(-1))
    t_policies = t_policies.mul(batch['tmask'].unsqueeze(-1)).sum(-2) - batch['pmask']

    # mask valid target values
    t_values = t_values.view(*batch['tmask'].size()[:2], -1)
    t_values = t_values.mul(batch['vmask'])

    return t_policies, t_values


def compose_losses(policies, values, log_selected_policies, advantages, value_targets, tmasks, vmasks, progress, entropy_regularization):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    losses = {}
    dcnt = tmasks.sum().item()

    turn_advantages = advantages.mul(tmasks).sum(-1, keepdim=True)

    losses['p'] = (-log_selected_policies * turn_advantages).sum()
    losses['v'] = ((values - value_targets) ** 2).mul(vmasks).sum() / 2

    entropy = dist.Categorical(logits=policies).entropy().mul(tmasks.sum(-1))
    losses['ent'] = entropy.sum()

    loss_base = losses['p'] + losses['v']
    losses['total'] = loss_base + entropy.mul(1 - progress * 0.9).sum() * -entropy_regularization

    return losses, dcnt


def vtrace_base(batch, model, hidden, args):
    t_policies, t_values = forward_prediction(model, hidden, batch, args['observation'])
    actions = batch['action']
    gmasks = batch['tmask'].sum(-1, keepdim=True)
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0

    log_selected_b_policies = F.log_softmax(batch['policy'], dim=-1).gather(-1, actions) * gmasks
    log_selected_t_policies = F.log_softmax(t_policies     , dim=-1).gather(-1, actions) * gmasks

    # thresholds of importance sampling
    log_rhos = log_selected_t_policies.detach() - log_selected_b_policies
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
    cs = torch.clamp(rhos, 0, clip_c_threshold)
    values_nograd = t_values.detach()

    if values_nograd.size(2) == 2:  # two player zerosum game
        values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=-1)
        if args['observation']:
            values_nograd = (values_nograd + values_nograd_opponent) / 2
        else:
            values_nograd = values_nograd + values_nograd_opponent

    values_nograd = values_nograd * gmasks + batch['return'] * (1 - gmasks)

    return t_policies, t_values, log_selected_t_policies, values_nograd, clipped_rhos, cs


def vtrace(batch, model, hidden, args):
    # IMPALA
    # https://github.com/deepmind/scalable_agent/blob/master/vtrace.py

    t_policies, t_values, log_selected_t_policies, values_nograd, clipped_rhos, cs = vtrace_base(batch, model, hidden, args)
    returns = batch['return']
    time_length = batch['vmask'].size(1)

    if args['return'] == 'MC':
        # VTrace with naive advantage
        value_targets = returns
        advantages = clipped_rhos * (returns - values_nograd)
    elif args['return'] == 'TD0':
        values_t_plus_1 = torch.cat([values_nograd[:, 1:], returns[:, -1:]], dim=1)
        deltas = clipped_rhos * (values_t_plus_1 - values_nograd)

        # compute Vtrace value target recursively
        vs_minus_v_xs = deque([deltas[:, -1]])
        for i in range(time_length - 2, -1, -1):
            vs_minus_v_xs.appendleft(deltas[:, i] + cs[:, i] * vs_minus_v_xs[0])
        vs_minus_v_xs = torch.stack(tuple(vs_minus_v_xs), dim=1)
        vs = vs_minus_v_xs + values_nograd

        # compute policy advantage
        value_targets = vs
        vs_t_plus_1 = torch.cat([vs[:, 1:], returns[:, -1:]], dim=1)
        advantages = clipped_rhos * (vs_t_plus_1 - values_nograd)
    elif args['return'] == 'TDLAMBDA':
        lmb = args['lambda']
        lambda_returns = deque([returns[:, -1]])
        for i in range(time_length - 2, -1, -1):
            lambda_returns.appendleft((1 - lmb) * values_nograd[:, i + 1] + lmb * lambda_returns[0])
        lambda_returns = torch.stack(tuple(lambda_returns), dim=1)

        value_targets = lambda_returns
        advantages = clipped_rhos * (value_targets - values_nograd)

    return compose_losses(
        t_policies, t_values, log_selected_t_policies, advantages, value_targets,
        batch['tmask'], batch['vmask'], batch['progress'], args['entropy_regularization'],
    )


class Batcher:
    def __init__(self, args, episodes):
        self.args = args
        self.episodes = episodes
        self.shutdown_flag = False

        if self.args['use_batcher_process']:
            self.workers = MultiProcessWorkers(
                self._worker, self._selector(), self.args['num_batchers'],
                buffer_length=3, num_receivers=2
            )
        else:
            self.workers = MultiThreadWorkers(self._worker, self._selector(), self.args['num_batchers'])

    def _selector(self):
        while True:
            yield [self.select_episode() for _ in range(self.args['batch_size'])]

    def _worker(self, conn, bid):
        print('started batcher %d' % bid)
        while not self.shutdown_flag:
            episodes = conn.recv()
            batch = make_batch(episodes, self.args)
            conn.send((batch, 1))
        print('finished batcher %d' % bid)

    def run(self):
        self.workers.start()

    def select_episode(self):
        while True:
            ep_idx = random.randrange(min(len(self.episodes), self.args['maximum_episodes']))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.args['maximum_episodes']
            if random.random() < accept_rate:
                break
        ep = self.episodes[ep_idx]
        turn_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])  # change start turn by sequence length
        st = random.randrange(turn_candidates)
        ed = min(st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        ep_minimum = {
            'args': ep['args'], 'reward': ep['reward'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'start': st, 'end': ed, 'total': ep['steps'],
        }
        return ep_minimum

    def batch(self):
        return self.workers.recv()

    def shutdown(self):
        self.shutdown_flag = True
        self.workers.shutdown()


class Trainer:
    def __init__(self, args, model):
        self.episodes = deque()
        self.args = args
        self.gpu = torch.cuda.device_count()
        self.model = model
        self.defalut_lr = 3e-8
        self.data_cnt_ema = self.args['batch_size'] * self.args['forward_steps']
        self.params = list(self.model.parameters())
        lr = self.defalut_lr * self.data_cnt_ema
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5) if len(self.params) > 0 else None
        self.steps = 0
        self.lock = threading.Lock()
        self.batcher = Batcher(self.args, self.episodes)
        self.updated_model = None, 0
        self.update_flag = False
        self.shutdown_flag = False

    def update(self):
        if len(self.episodes) < self.args['minimum_episodes']:
            return None, 0  # return None before training
        self.update_flag = True
        while True:
            time.sleep(0.1)
            model, steps = self.recheck_update()
            if model is not None:
                break
        return model, steps

    def report_update(self, model, steps):
        self.lock.acquire()
        self.update_flag = False
        self.updated_model = model, steps
        self.lock.release()

    def recheck_update(self):
        self.lock.acquire()
        flag = self.update_flag
        self.lock.release()
        return (None, -1) if flag else self.updated_model

    def shutdown(self):
        self.shutdown_flag = True
        self.batcher.shutdown()

    def train(self):
        if self.optimizer is None:  # non-parametric model
            print()
            return

        batch_cnt, data_cnt, loss_sum = 0, 0, {}
        train_model = self.model
        if self.gpu:
            if self.gpu > 1:
                train_model = nn.DataParallel(self.model)
            train_model.cuda()
        train_model.train()

        while data_cnt == 0 or not (self.update_flag or self.shutdown_flag):
            # episodes were only tuple of arrays
            batch = to_gpu_or_not(self.batcher.batch(), self.gpu)
            batch_size = batch['value'].size(0)
            player_count = batch['value'].size(2)
            hidden = to_gpu_or_not(self.model.init_hidden([batch_size, player_count]), self.gpu)

            losses, dcnt = vtrace(batch, train_model, hidden, self.args)

            self.optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.params, 4.0)
            self.optimizer.step()

            batch_cnt += 1
            data_cnt += dcnt
            for k, l in losses.items():
                loss_sum[k] = loss_sum.get(k, 0.0) + l.item()

            self.steps += 1

        print('loss = %s' % ' '.join([k + ':' + '%.3f' % (l / data_cnt) for k, l in loss_sum.items()]))

        self.data_cnt_ema = self.data_cnt_ema * 0.8 + data_cnt / (1e-2 + batch_cnt) * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.defalut_lr * self.data_cnt_ema / (1 + self.steps * 1e-5)
        self.model.cpu()
        self.model.eval()
        return copy.deepcopy(self.model)

    def run(self):
        print('waiting training')
        while not self.shutdown_flag:
            if len(self.episodes) < self.args['minimum_episodes']:
                time.sleep(1)
                continue
            if self.steps == 0:
                self.batcher.run()
                print('started training')
            model = self.train()
            self.report_update(model, self.steps)
        print('finished training')


class Learner:
    def __init__(self, args):
        self.args = args
        random.seed(args['seed'])

        self.env = make_env(args['env'])
        eval_modify_rate = (args['update_episodes'] ** 0.85) / args['update_episodes']
        self.eval_rate = max(args['eval_rate'], eval_modify_rate)
        self.shutdown_flag = False

        # trained datum
        self.model_era = self.args['restart_epoch']
        self.model_class = self.env.net() if hasattr(self.env, 'net') else Model
        train_model = self.model_class(self.env, args)
        if self.model_era == 0:
            self.model = RandomModel(self.env)
        else:
            self.model = train_model
            self.model.load_state_dict(torch.load(self.model_path(self.model_era)), strict=False)

        # generated datum
        self.num_episodes = 0

        # evaluated datum
        self.results = {}
        self.num_results = 0

        # multiprocess or remote connection
        self.workers = Workers(args)

        # thread connection
        self.trainer = Trainer(args, train_model)

    def shutdown(self):
        self.shutdown_flag = True
        self.trainer.shutdown()
        self.workers.shutdown()
        for thread in self.threads:
            thread.join()

    def model_path(self, model_id):
        return os.path.join('models', str(model_id) + '.pth')

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_era += 1
        self.model = model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), self.model_path(self.model_era))

    def feed_episodes(self, episodes):
        # store generated episodes
        self.trainer.episodes.extend([e for e in episodes if e is not None])
        while len(self.trainer.episodes) > self.args['maximum_episodes']:
            self.trainer.episodes.popleft()

    def feed_results(self, results):
        # store evaluation results
        for model_id, reward in results:
            if reward is None:
                continue
            if model_id not in self.results:
                self.results[model_id] = {}
            if reward not in self.results[model_id]:
                self.results[model_id][reward] = 0
            self.results[model_id][reward] += 1

    def update(self):
        # call update to every component
        if self.model_era not in self.results:
            print('win rate = Nan (0)')
        else:
            distribution = self.results[self.model_era]
            results = {k: distribution[k] for k in sorted(distribution.keys(), reverse=True)}
            # output evaluation results
            n, win = 0, 0.0
            for r, cnt in results.items():
                n += cnt
                win += (r + 1) / 2 * cnt
            print('win rate = %.3f (%.1f / %d)' % (win / n, win, n))
        model, steps = self.trainer.update()
        if model is None:
            model = self.model
        self.update_model(model, steps)

    def server(self):
        # central conductor server
        # returns as list if getting multiple requests as list
        print('started server')
        prev_update_episodes = self.args['minimum_episodes']
        while True:
            # no update call before storings minimum number of episodes + 1 age
            next_update_episodes = prev_update_episodes + self.args['update_episodes']
            while not self.shutdown_flag and self.num_episodes < next_update_episodes:
                conn, (req, data) = self.workers.recv()
                multi_req = isinstance(data, list)
                if not multi_req:
                    data = [data]
                send_data = []

                if req == 'args':
                    for _ in data:
                        args = {'model_id': {}}

                        # decide role
                        if self.num_results < self.eval_rate * self.num_episodes:
                            args['role'] = 'e'
                        else:
                            args['role'] = 'g'

                        if args['role'] == 'g':
                            # genatation configuration
                            args['player'] = self.env.players()[self.num_episodes % len(self.env.players())]
                            for p in self.env.players():
                                args['model_id'][p] = self.model_era
                            self.num_episodes += 1
                            if self.num_episodes % 100 == 0:
                                print(self.num_episodes, end=' ', flush=True)

                        elif args['role'] == 'e':
                            # evaluation configuration
                            args['player'] = self.env.players()[self.num_results % len(self.env.players())]
                            for p in self.env.players():
                                if p == args['player']:
                                    args['model_id'][p] = self.model_era
                                else:
                                    args['model_id'][p] = -1
                            self.num_results += 1

                        send_data.append(args)

                elif req == 'episode':
                    # report generated episodes
                    self.feed_episodes(data)
                    send_data = [None] * len(data)

                elif req == 'result':
                    # report evaluation results
                    self.feed_results(data)
                    send_data = [None] * len(data)

                elif req == 'model':
                    for model_id in data:
                        if model_id == self.model_era:
                            model = self.model
                        else:
                            try:
                                model = self.model_class(self.env, self.args)
                                model.load_state_dict(torch.load(self.model_path(model_id)), strict=False)
                            except:
                                # return latest model if failed to load specified model
                                pass
                        send_data.append(model)

                if not multi_req and len(send_data) == 1:
                    send_data = send_data[0]
                self.workers.send(conn, send_data)
            prev_update_episodes = next_update_episodes
            self.update()
        print('finished server')

    def entry_server(self):
        port = 9999
        print('started entry server %d' % port)
        conn_acceptor = accept_socket_connections(port=port, timeout=0.3)
        while not self.shutdown_flag:
            conn = next(conn_acceptor)
            if conn is not None:
                entry_args = conn.recv()
                print('accepted entry from %s!' % entry_args['host'])
                args = copy.deepcopy(self.args)
                args['worker'] = entry_args
                conn.send(args)
                conn.close()
        print('finished entry server')

    def run(self):
        try:
            # open threads
            self.threads = [threading.Thread(target=self.trainer.run)]
            if self.args['remote']:
                self.threads.append(threading.Thread(target=self.entry_server))
            for thread in self.threads:
                thread.start()
            # open generator, evaluator
            self.workers.run()
            self.server()

        finally:
            self.shutdown()


def train_main(args):
    train_args = args['train_args']
    train_args['remote'] = False

    env_args = args['env_args']
    train_args['env'] = env_args

    prepare_env(env_args)  # preparing environment is needed in stand-alone mode
    learner = Learner(train_args)
    learner.run()


def train_server_main(args):
    train_args = args['train_args']
    train_args['remote'] = True

    env_args = args['env_args']
    train_args['env'] = env_args

    learner = Learner(train_args)
    learner.run()
