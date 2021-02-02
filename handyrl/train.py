# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

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
from .util import map_r, bimap_r, trimap_r, rotate
from .model import to_torch, to_gpu_or_not, RandomModel
from .model import SimpleConv2DModel as DefaultModel
from .losses import compute_target
from .connection import MultiProcessWorkers
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
        Basic data shape is (B, T, P, ...) .
        (B is batch size, T is time length, P is player count)
    """

    obss, datum = [], []

    def replace_none(a, b):
        return a if a is not None else b

    for ep in episodes:
        # target player and turn index
        moments_ = sum([pickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
        moments = moments_[ep['start'] - ep['base']:ep['end'] - ep['base']]
        players = list(moments[0]['observation'].keys())

        obs_zeros = map_r(moments[0]['observation'][moments[0]['turn']], lambda o: np.zeros_like(o))  # template for padding
        if args['observation']:
            # replace None with zeros
            obs = [[replace_none(m['observation'][player], obs_zeros) for player in players] for m in moments]
        else:
            obs = [[m['observation'][m['turn']]] for m in moments]
        obs = rotate(obs)  # (T, P, ..., ...) -> (P, ..., T, ...)
        obs = rotate(obs)  # (P, ..., T, ...) -> (..., T, P, ...)
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))

        # datum that is not changed by training configuration
        p = np.array([[m['policy'][m['turn']]] for m in moments])
        v = np.array([replace_none(m['value'][m['turn']], [0, 0]) for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        rew = np.array([[replace_none(m['reward'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        ret = np.array([[replace_none(m['return'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        oc = np.array([ep['outcome'][player] for player in players], dtype=np.float32).reshape(1, len(players), -1)

        emask = np.ones((len(moments), 1, 1), dtype=np.float32)  # episode mask
        amask = np.array([[m['action_mask'][m['turn']]] for m in moments])
        tmask = np.array([[[m['policy'][player] is not None] for player in players] for m in moments], dtype=np.float32)
        omask = np.array([[[m['value'][player] is not None] for player in players] for m in moments], dtype=np.float32)

        act = np.array([[m['action']] for m in moments], dtype=np.int64)[..., np.newaxis]

        progress = np.arange(ep['start'], ep['end'], dtype=np.float32)[..., np.newaxis] / ep['total']

        # pad each array if step length is short
        if len(tmask) < args['forward_steps']:
            pad_len = args['forward_steps'] - len(tmask)
            obs = map_r(obs, lambda o: np.pad(o, [(0, pad_len)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0))
            p = np.pad(p, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            v = np.concatenate([v, np.tile(oc, [pad_len, 1, 1])])
            act = np.concatenate([act, [[[random.randrange(len(p[0]))]] for _ in range(pad_len)]])
            rew = np.pad(rew, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            ret = np.pad(ret, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            emask = np.pad(emask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=1)
            tmask = np.pad(tmask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=1)
            omask = np.pad(omask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=1)
            amask = np.pad(amask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=1e32)
            progress = np.pad(progress, [(0, pad_len), (0, 0)], 'constant', constant_values=1)

        obss.append(obs)
        datum.append((p, v, act, oc, rew, ret, emask, tmask, omask, amask, progress))

    p, v, act, oc, rew, ret, emask, tmask, omask, amask, progress = zip(*datum)

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    p = to_torch(np.array(p))
    v = to_torch(np.array(v))
    act = to_torch(np.array(act))
    oc = to_torch(np.array(oc))
    rew = to_torch(np.array(rew))
    ret = to_torch(np.array(ret))
    emask = to_torch(np.array(emask))
    tmask = to_torch(np.array(tmask))
    omask = to_torch(np.array(omask))
    amask = to_torch(np.array(amask))
    progress = to_torch(np.array(progress))

    return {
        'observation': obs,
        'policy': p, 'value': v,
        'action': act, 'outcome': oc,
        'reward': rew, 'return': ret,
        'episode_mask': emask,
        'turn_mask': tmask, 'observation_mask': omask,
        'action_mask': amask,
        'progress': progress,
    }


def forward_prediction(model, hidden, batch, obs_mode):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: batch outputs of newral network
    """

    observations = batch['observation']  # (B, T, P, ...)

    if hidden is None:
        # feed-forward neural network
        obs = map_r(observations, lambda o: o.view(-1, *o.size()[3:]))
        outputs = model(obs, None)
    else:
        # sequential computation with RNN
        outputs = {}
        for t in range(batch['turn_mask'].size(1)):
            obs = map_r(observations, lambda o: o[:, t].reshape(-1, *o.size()[3:]))  # (..., B * P, ...)
            action = batch['action'][:, t]
            #omask_ = batch['observation_mask'][:, t]
            #omask_ = omask_.sum(-1, keepdim=True)  # common hidden state
            #omask = map_r(hidden, lambda h: omask_.view(*h.size()[:2], *([1] * (len(h.size()) - 2))))
            #hidden_ = bimap_r(hidden, omask, lambda h, m: h * m)  # (..., B, P, ...)
            #if obs_mode:
            #    hidden_ = map_r(hidden_, lambda h: h.view(-1, *h.size()[2:]))  # (..., B * P, ...)
            #else:
            #    hidden_ = map_r(hidden_, lambda h: h.sum(1))  # (..., B * 1, ...)
            hidden_ = hidden
            outputs_ = model(obs, hidden_, action)
            for k, o in outputs_.items():
                if k == 'hidden':
                    next_hidden = outputs_['hidden']
                else:
                    outputs[k] = outputs.get(k, []) + [o]
            #next_hidden = bimap_r(next_hidden, hidden, lambda nh, h: nh.view(h.size(0), -1, *h.size()[2:]))  # (..., B, P or 1, ...)
            #hidden = trimap_r(hidden, next_hidden, omask, lambda h, nh, m: h * (1 - m) + nh * m)
            hidden = next_hidden
        outputs = {k: torch.stack(o, dim=1) for k, o in outputs.items() if o[0] is not None}

    for k, o in outputs.items():
        o = o.view(*batch['turn_mask'].size()[:2], -1, o.size(-1))
        if k == 'policy':
            # gather turn player's policies
            outputs[k] = o.mul(batch['turn_mask']).sum(2, keepdim=True)# - batch['action_mask']
        else:
            # mask valid target values and cumulative rewards
            o = o.view(*batch['turn_mask'].size()[:2], -1, 1)
            outputs[k] = o.mul(batch['observation_mask'])

    return outputs


def compose_losses(outputs, log_selected_policies, total_advantages, targets, batch, args):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    tmasks = batch['turn_mask']
    omasks = batch['observation_mask']

    losses = {}
    dcnt = tmasks.sum().item()
    turn_advantages = total_advantages.mul(tmasks).sum(2, keepdim=True)

    losses['p'] = (-log_selected_policies * turn_advantages).sum()
    losses['pp'] = (F.softmax(batch['policy'], -1) * (F.log_softmax(batch['policy'], -1) - F.log_softmax(outputs['policy'], -1))).sum(-1, keepdim=True).mul(tmasks).sum()
    if 'value' in outputs:
        losses['v'] = ((outputs['value'] - targets['value']) ** 2).mul(omasks).sum() / 2
        losses['pv'] = ((outputs['value'] - batch['outcome']) ** 2).mul(omasks).sum() / 2
    if 'return' in outputs:
        losses['r'] = F.smooth_l1_loss(outputs['return'], targets['return'], reduction='none').mul(omasks).sum()

    entropy = dist.Categorical(logits=outputs['policy'] - batch['action_mask']).entropy().mul(tmasks.sum(-1))
    losses['ent'] = entropy.sum()

    planning_weight = 1
    reinforce_loss = losses['p'] + losses.get('v', 0) + losses.get('r', 0)
    planning_loss = losses['pp'] + losses.get('pv', 0)
    base_loss = (1 - planning_weight) * reinforce_loss + planning_weight * planning_loss
    entropy_loss = entropy.mul(1 - batch['progress'] * (1 - args['entropy_regularization_decay'])).sum() * -args['entropy_regularization']
    losses['total'] = base_loss + entropy_loss

    return losses, dcnt


def compute_loss(batch, model, hidden, args):
    outputs = forward_prediction(model, hidden, batch, args['observation'])
    actions = batch['action']
    emasks = batch['episode_mask']
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0

    log_selected_b_policies = F.log_softmax(batch['policy']  , dim=-1).gather(-1, actions) * emasks
    log_selected_t_policies = F.log_softmax(outputs['policy'], dim=-1).gather(-1, actions) * emasks

    # thresholds of importance sampling
    log_rhos = log_selected_t_policies.detach() - log_selected_b_policies
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
    cs = torch.clamp(rhos, 0, clip_c_threshold)
    outputs_nograd = {k: o.detach() for k, o in outputs.items()}

    if 'value' in outputs_nograd:
        values_nograd = outputs_nograd['value']
        if values_nograd.size(2) == 2:  # two player zerosum game
            values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
            if args['observation']:
                values_nograd = (values_nograd + values_nograd_opponent) / 2
            else:
                values_nograd = values_nograd + values_nograd_opponent
        outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)

    # compute targets and advantage
    targets = {}
    advantages = {}

    value_args = outputs_nograd.get('value', None), batch['outcome'], None, args['lambda'], 1, clipped_rhos, cs
    return_args = outputs_nograd.get('return', None), batch['return'], batch['reward'], args['lambda'], args['gamma'], clipped_rhos, cs

    targets['value'], advantages['value'] = compute_target(args['value_target'], *value_args)
    targets['return'], advantages['return'] = compute_target(args['value_target'], *return_args)

    if args['policy_target'] != args['value_target']:
        _, advantages['value'] = compute_target(args['policy_target'], *value_args)
        _, advantages['return'] = compute_target(args['policy_target'], *return_args)

    # compute policy advantage
    total_advantages = clipped_rhos * sum(advantages.values())

    return compose_losses(outputs, log_selected_t_policies, total_advantages, targets, batch, args)


class Batcher:
    def __init__(self, args, episodes):
        self.args = args
        self.episodes = episodes
        self.shutdown_flag = False

        self.workers = MultiProcessWorkers(
            self._worker, self._selector(), self.args['num_batchers'],
            buffer_length=3, num_receivers=2
        )

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
        #turn_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])  # change start turn by sequence length
        turn_candidates = ep['steps']
        st = random.randrange(turn_candidates)
        ed = min(st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        ep_minimum = {
            'args': ep['args'], 'outcome': ep['outcome'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'start': st, 'end': ed, 'total': ep['steps']
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

            losses, dcnt = compute_loss(batch, train_model, hidden, self.args)

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
        self.model_class = self.env.net() if hasattr(self.env, 'net') else DefaultModel
        train_model = self.model_class(self.env, args)
        if self.model_era == 0:
            self.model = RandomModel(self.env)
        else:
            self.model = train_model
            self.model.load_state_dict(torch.load(self.model_path(self.model_era)), strict=False)

        # generated datum
        self.generation_results = {}
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

    def latest_model_path(self):
        return os.path.join('models', 'latest.pth')

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_era += 1
        self.model = model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), self.model_path(self.model_era))
        torch.save(model.state_dict(), self.latest_model_path())

    def feed_episodes(self, episodes):
        # analyze generated episodes
        for episode in episodes:
            if episode is None:
                continue
            for p in episode['args']['player']:
                model_id = episode['args']['model_id'][p]
                outcome = episode['outcome'][p]
                n, r, r2 = self.generation_results.get(model_id, (0, 0, 0))
                self.generation_results[model_id] = n + 1, r + outcome, r2 + outcome ** 2

        # store generated episodes
        self.trainer.episodes.extend([e for e in episodes if e is not None])
        while len(self.trainer.episodes) > self.args['maximum_episodes']:
            self.trainer.episodes.popleft()

    def feed_results(self, results):
        # store evaluation results
        for result in results:
            if result is None:
                continue
            for p in result['args']['player']:
                model_id = result['args']['model_id'][p]
                res = result['result'][p]
                n, r, r2 = self.results.get(model_id, (0, 0, 0))
                self.results[model_id] = n + 1, r + res, r2 + res ** 2

    def update(self):
        # call update to every component
        print()
        print('epoch %d' % self.model_era)

        if self.model_era not in self.results:
            print('win rate = Nan (0)')
        else:
            n, r, r2 = self.results[self.model_era]
            mean = r / (n + 1e-6)
            print('win rate = %.3f (%.1f / %d)' % ((mean + 1) / 2, (r + n) / 2, n))

        if self.model_era not in self.generation_results:
            print('generation stats = Nan (0)')
        else:
            n, r, r2 = self.generation_results[self.model_era]
            mean = r / (n + 1e-6)
            std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
            print('generation stats = %.3f +- %.3f' % (mean, std))

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
                            args['player'] = self.env.players()
                            for p in self.env.players():
                                if p in args['player']:
                                    args['model_id'][p] = self.model_era
                                else:
                                    args['model_id'][p] = -1
                            self.num_episodes += 1
                            if self.num_episodes % 100 == 0:
                                print(self.num_episodes, end=' ', flush=True)

                        elif args['role'] == 'e':
                            # evaluation configuration
                            args['player'] = [self.env.players()[self.num_results % len(self.env.players())]]
                            for p in self.env.players():
                                if p in args['player']:
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
                        model = self.model
                        if model_id != self.model_era:
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
