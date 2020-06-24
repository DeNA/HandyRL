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
import signal
import bz2
import pickle
import yaml
from collections import deque

import numpy as np
import tensorflow as tf

import environment as gym
from model import to_tensor, to_numpy, to_gpu_or_not, softmax, RandomModel
from model import DuelingNet as Model
from connection import MultiProcessWorkers, MultiThreadWorkers
from connection import accept_socket_connections
from worker import Workers


def make_batch(episodes, args):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes
        args (dict): training configuration

    Returns:
        dict: input and target tensors

    Note:
        Basic data shape is (T, B, P, ...) .
        (T is time length, B is batch size, P is player count)
    """

    datum = []  # obs, act, p, v, ret, adv, len
    steps = args['forward_steps']

    for ep in episodes:
        ep = pickle.loads(bz2.decompress(ep))

        # target player and turn index
        players = sorted([player for player in ep['value'].keys() if player >= 0])
        ep_train_length = len(ep['turn'])
        turn_candidates = 1 + max(0, ep_train_length - args['forward_steps'])  # change start turn by sequence length
        st = random.randrange(turn_candidates)
        ed = min(st + steps, len(ep['turn']))

        obs_sample = ep['observation'][ep['turn'][st]][st]
        if args['observation']:
            obs_zeros = tuple(np.zeros_like(o) for o in obs_sample)  # template for padding
            # transpose observation from (P, T, tuple) to (tuple, T, P)
            obs = []
            for i, _ in enumerate(obs_zeros):
                obs.append([])
                for t in range(st, ed):
                    obs[-1].append([])
                    for player in players:
                        obs[-1][-1].append(ep['observation'][player][t][i] if ep['observation'][player][t] is not None else obs_zeros[i])
        else:
            obs = tuple([[ep['observation'][ep['turn'][t]][t][i]] for t in range(st, ed)] for i, _ in enumerate(obs_sample))

        obs = tuple(np.array(o) for o in obs)

        # datum that is not changed by training configuration
        v = np.array(
            [[ep['value'][player][t] or 0 for player in players] for t in range(st, ed)],
            dtype=np.float32
        ).reshape(-1, len(players))
        tmsk = np.eye(len(players))[ep['turn'][st:ed]]
        pmsk = np.array(ep['pmask'][st:ed])
        vmsk = np.ones_like(tmsk) if args['observation'] else tmsk

        act = np.array(ep['action'][st:ed]).reshape(-1, 1)
        p = np.array(ep['policy'][st:ed])
        progress = np.arange(st, ed, dtype=np.float32) / len(ep['turn'])

        traj_steps = len(tmsk)
        ret = np.array(ep['reward'], dtype=np.float32).reshape(1, -1)

        # pad each array if step length is short
        if traj_steps < steps:
            pad_len = steps - traj_steps
            obs = tuple(np.pad(o, [(0, pad_len)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0) for o in obs)
            v = np.concatenate([v, np.tile(ret, [pad_len, 1])])
            tmsk = np.pad(tmsk, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            pmsk = np.pad(pmsk, [(0, pad_len), (0, 0)], 'constant', constant_values=1e32)
            vmsk = np.pad(vmsk, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            act = np.pad(act, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            p = np.pad(p, [(0, pad_len), (0, 0)], 'constant', constant_values=0)
            progress = np.pad(progress, [(0, pad_len)], 'constant', constant_values=1)

        datum.append((obs, tmsk, pmsk, vmsk, act, p, v, ret, progress))

    obs, tmsk, pmsk, vmsk, act, p, v, ret, progress = zip(*datum)

    obs = tuple(to_tensor(o, transpose=True) for o in zip(*obs))
    tmsk = to_tensor(tmsk, transpose=True)
    pmsk = to_tensor(pmsk, transpose=True)
    vmsk = to_tensor(vmsk, transpose=True)
    act = to_tensor(act, transpose=True)
    p = to_tensor(p, transpose=True)
    v = to_tensor(v, transpose=True)
    ret = to_tensor(ret, transpose=True)
    progress = to_tensor(progress, transpose=True)

    return {
        'observation': obs, 'tmask': tmsk, 'pmask': pmsk, 'vmask': vmsk,
        'action': act, 'policy': p, 'value': v, 'return': ret, 'progress': progress,
    }


def forward_prediction(model, hidden, batch):
    """Forward calculation via neural network

    Args:
        model (tensorflow.keras.Model): neural network
        hidden: initial hidden state
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: calculated policy and value
    """

    observations = batch['observation']

    if hidden is None:
        # feed-forward neural network
        obs = tuple(tf.reshape(o, [-1, *(o.shape[3:])]) for o in observations)
        t_policies, t_values, _ = model(obs, None)
    else:
        # sequential computation with RNN
        time_length = observations[0].shape[0]
        bmasks = batch['tmask'] + batch['vmask']
        bmasks = tuple(tf.reshape(bmasks, [time_length, 1, bmasks.shape[1], bmasks.shape[2],
            *[1 for _ in range(len(h.shape) - 3)]]) for h in hidden)

        t_policies, t_values = [], []
        for t in range(time_length):
            bmask = tuple(m[t] for m in bmasks)
            obs = tuple(tf.reshape(o[t], [-1, o.shape[1], *(o.shape[3:])]) for o in observations)
            hidden = tuple(h * bmask[i] for i, h in enumerate(hidden))
            if observations[0].shape[2] == 1:
                hid = tuple(tf.reduce_sum(h, 2) for h in hidden)
            else:
                hid = tuple(tf.reshape(h, [-1, h.shape[1] * h.shape[2], *(h.shape[3:])]) for h in hidden)
            t_policy, t_value, next_hidden = model(obs, hid)
            t_policies.append(t_policy)
            t_values.append(t_value)
            next_hidden = tuple(tf.reshape(h, [h.shape[0], -1, observations[0].shape[2], *(h.shape[2:])]) for h in next_hidden)
            hidden = tuple(hidden[i] * (1 - bmask[i]) + h * bmask[i] for i, h in enumerate(next_hidden))
        t_policies = tf.stack(t_policies)
        t_values = tf.stack(t_values)

    # gather turn player's policies
    t_policies = tf.reshape(t_policies, [*observations[0].shape[:3], t_policies.shape[-1]])
    t_policies = t_policies * tf.expand_dims(batch['tmask'], -1)
    t_policies = tf.reduce_sum(t_policies, -2) - batch['pmask']

    # mask valid target values
    t_values = tf.reshape(t_values, [*(observations[0].shape[:3])])
    t_values = t_values * batch['vmask']

    return t_policies, t_values


def compose_losses(policies, values, log_selected_policies, advantages, value_targets, tmasks, vmasks, progress):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    losses = {}
    dcnt = tmasks.sum()

    turn_advantages = tf.reduce_sum(advantages * tmasks, -1, keepdims=True)

    losses['p'] = tf.reduce_sum(-log_selected_policies * turn_advantages)
    losses['v'] = tf.reduce_sum(((values - value_targets) ** 2) * vmasks) / 2

    def compute_entropy(p):
        return -tf.reduce_sum(tf.nn.softmax(p) * tf.nn.log_softmax(p), -1)

    entropy = compute_entropy(policies) * tf.reduce_sum(tmasks, -1)
    losses['ent'] = tf.reduce_sum(entropy)

    losses['total'] = losses['p'] + losses['v'] + tf.reduce_sum(entropy * (1 - progress * 0.9)) * -3e-1

    return losses, dcnt


def vtrace_base(batch, model, hidden, args):
    t_policies, t_values = forward_prediction(model, hidden, batch)
    actions = batch['action']
    gmasks = tf.reduce_sum(batch['tmask'], -1, keepdims=True)
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0

    log_selected_b_policies = tf.gather(tf.nn.log_softmax(batch['policy']), actions, axis=-1, batch_dims=2) * gmasks
    log_selected_t_policies = tf.gather(tf.nn.log_softmax(t_policies     ), actions, axis=-1, batch_dims=2) * gmasks

    # thresholds of importance sampling
    log_rhos = tf.stop_gradient(log_selected_t_policies) - log_selected_b_policies
    rhos = tf.exp(log_rhos)
    clipped_rhos = tf.clip_by_value(rhos, 0, clip_rho_threshold)
    cs = tf.clip_by_value(rhos, 0, clip_c_threshold)
    values_nograd = tf.stop_gradient(t_values)

    if values_nograd.shape[2] == 2:  # two player zerosum game
        values_nograd_opponent = -tf.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], -1)
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
    time_length = batch['vmask'].shape[0]

    if args['return'] == 'MC':
        # VTrace with naive advantage
        value_targets = returns
        advantages = clipped_rhos * (returns - values_nograd)
    elif args['return'] == 'TD0':
        values_t_plus_1 = tf.concat([values_nograd[1:], returns])
        deltas = clipped_rhos * (values_t_plus_1 - values_nograd)

        # compute Vtrace value target recursively
        vs_minus_v_xs = deque([deltas[-1]])
        for i in range(time_length - 2, -1, -1):
            vs_minus_v_xs.appendleft(deltas[i] + cs[i] * vs_minus_v_xs[0])
        vs_minus_v_xs = tf.stack(tuple(vs_minus_v_xs))
        vs = vs_minus_v_xs + values_nograd

        # compute policy advantage
        value_targets = vs
        vs_t_plus_1 = tf.concat([vs[1:], returns])
        advantages = clipped_rhos * (vs_t_plus_1 - values_nograd)
    elif args['return'] == 'TDLAMBDA':
        lmb = 0.7
        lambda_returns = deque([returns[-1]])
        for i in range(time_length - 1, 0, -1):
            lambda_returns.appendleft((1 - lmb) * values_nograd[i] + lmb * lambda_returns[0])
        lambda_returns = tf.stack(tuple(lambda_returns))

        value_targets = lambda_returns
        advantages = clipped_rhos * (value_targets - values_nograd)

    return compose_losses(
        t_policies, t_values, log_selected_t_policies, advantages, value_targets,
        batch['tmask'], batch['vmask'], batch['progress']
    )


class Batcher:
    def __init__(self, args, episodes):
        self.args = args
        self.episodes = episodes
        self.shutdown_flag = False

        def selector():
            while True:
                yield self.select_episode()

        def worker(conn, bid):
            print('started batcher %d' % bid)
            episodes = []
            while not self.shutdown_flag:
                ep = conn.recv()
                episodes.append(ep)
                if len(episodes) >= self.args['batch_size']:
                    batch = make_batch(episodes, self.args)
                    conn.send((batch, len(episodes)))
                    episodes = []
            print('finished batcher %d' % bid)

        # self.workers = MultiProcessWorkers(worker, selector(), self.args['num_batchers'], buffer_length=self.args['batch_size'] * 3, num_receivers=2)
        self.workers = MultiThreadWorkers(worker, selector(), self.args['num_batchers'])

    def run(self):
        self.workers.start()

    def select_episode(self):
        while True:
            ep_idx = random.randrange(min(len(self.episodes), self.args['maximum_episodes']))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.args['maximum_episodes']
            if random.random() < accept_rate:
                return self.episodes[ep_idx]

    def batch(self):
        return self.workers.recv()

    def shutdown(self):
        self.shutdown_flag = True
        self.workers.shutdown()


class Trainer:
    def __init__(self, args, model):
        self.episodes = deque()
        self.args = args
        self.model = model
        self.defalut_lr = 3e-8
        batch_data_cnt = self.args['batch_size'] * self.args['forward_steps']
        lr = self.defalut_lr * batch_data_cnt
        self.optimizer = tf.keras.optimizers.Adam(lr) if len(self.model.trainable_variables) > 0 else None
        self.steps = 0
        self.lock = threading.Lock()
        self.batcher = Batcher(self.args, self.episodes)
        self.shutdown_flag = False

        # temporal datum
        self.data_cnt = 0
        self.loss_sum = {}

    def update(self):
        if len(self.episodes) < self.args['minimum_episodes']:
            return None, 0  # return None before training
        self.lock.acquire()
        weights = self.model.get_weights()
        steps = self.steps
        data_cnt = self.data_cnt
        loss_sum = copy.deepcopy(self.loss_sum)
        self.data_cnt, self.loss_sum = 0, {}  # clear temporal stats
        self.lock.release()
        if data_cnt > 0:
            print('loss = %s' % ' '.join([k + ':' + '%.3f' % (l / data_cnt) for k, l in loss_sum.items()]))
        return weights, steps

    def shutdown(self):
        self.shutdown_flag = True
        self.batcher.shutdown()

    def train(self):
        print('started training')
        train_model = self.model

        @tf.function
        def train_model_fn(x, h):
            return train_model(x, h)

        while not self.shutdown_flag:
            # episodes were only tuple of arrays
            batch = self.batcher.batch()
            batch_size = batch['value'].shape[1]
            player_count = batch['value'].shape[2]
            hidden = self.model.init_hidden([batch_size, player_count])

            self.lock.acquire()
            with tf.GradientTape() as tape:
                losses, dcnt = vtrace(batch, train_model_fn, hidden, self.args)

            gradients = tape.gradient(losses['total'], self.model.trainable_variables)
            #nn.utils.clip_grad_norm_(self.params, 4.0)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.data_cnt += dcnt
            for k, l in losses.items():
                self.loss_sum[k] = self.loss_sum.get(k, 0.0) + float(to_numpy(l))
            self.steps += 1
            self.lock.release()
        print('finished training')

    def run(self):
        print('waiting training')
        while (not self.shutdown_flag) and len(self.episodes) < self.args['minimum_episodes']:
            time.sleep(0.2)
            continue

        if self.optimizer is not None:
            self.batcher.run()
            self.train()


class Learner:
    def __init__(self, args):
        self.args = args
        random.seed(args['seed'])
        self.env = gym.make()
        self.shutdown_flag = False

        # trained datum
        self.model_era = 0
        self.model_class = self.env.net() if hasattr(self.env, 'net') else Model
        self.model = self.model_class(self.env, args)
        self.model.inference(self.env.observation())
        tf.saved_model.save(self.model, self.model_path(self.model_era))

        # generated datum
        self.num_episodes = 0

        # evaluated datum
        self.results = {}
        self.num_results = 0

        # multiprocess or remote connection
        self.workers = Workers(args)

        # thread connection
        train_model = self.model_class(self.env, args)
        train_model.inference(self.env.observation())
        self.trainer = Trainer(args, train_model)

    def shutdown(self):
        self.shutdown_flag = True
        self.trainer.shutdown()
        self.workers.shutdown()
        for thread in self.threads:
            thread.join()

    def model_path(self, model_id):
        return os.path.join('models', str(model_id))

    def update_model(self, weights, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_era += 1
        if weights is not None:
            self.model.set_weights(weights)
        os.makedirs('models', exist_ok=True)
        tf.saved_model.save(self.model, self.model_path(self.model_era))

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
        weights, steps = self.trainer.update()
        self.update_model(weights, steps)

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
                if req == 'gargs':
                    # genatation configuration
                    for _ in data:
                        args = {
                            'episode_id': self.num_episodes,
                            'player': self.num_episodes % 2,
                            'model_id': {}
                        }
                        num_congress = int(1 + np.log2(self.model_era + 1)) if self.args['congress'] else 1
                        for p in range(2):
                            if p == args['player']:
                                args['model_id'][p] = [self.model_era]
                            else:
                                args['model_id'][p] = [self.model_era]  # [random.randrange(self.model_era + 1) for _ in range(num_congress)]
                        send_data.append(args)

                        self.num_episodes += 1
                        if self.num_episodes % 100 == 0:
                            print(self.num_episodes, end=' ', flush=True)
                elif req == 'eargs':
                    # evaluation configuration
                    for _ in data:
                        args = {
                            'model_id': self.model_era,
                            'player': self.num_results % 2,
                        }
                        send_data.append(args)
                        self.num_results += 1
                elif req == 'episode':
                    # report generated episodes
                    self.feed_episodes(data)
                    send_data = [True] * len(data)
                elif req == 'result':
                    # report evaluation results
                    self.feed_results(data)
                    send_data = [True] * len(data)
                elif req == 'model':
                    send_data = []
                    for model_id in data:
                        if model_id == self.model_era:
                            model = self.model
                        else:
                            try:
                                model = tf.keras.models.load_model()
                            except:
                                # return latest model if failed to load specified model
                                pass
                        send_data.append((model.get_weights()))
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
        total_gids, total_eids, worker_cnt = [], [], 0
        while not self.shutdown_flag:
            conn = next(conn_acceptor)
            if conn is not None:
                entry_args = conn.recv()
                print('accepted entry from %s!' % entry_args['host'])
                gids, eids = [], []
                # divide workers into generator/worker
                for _ in range(entry_args['num_process']):
                    if len(total_gids) * self.args['eworker_rate'] < len(total_eids) - 1:
                        gids.append(worker_cnt)
                        total_gids.append(worker_cnt)
                    else:
                        eids.append(worker_cnt)
                        total_eids.append(worker_cnt)
                    worker_cnt += 1
                args = copy.deepcopy(self.args)
                args['worker'] = entry_args
                args['gids'], args['eids'] = gids, eids
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
                thread.daemon = True
                thread.start()
            # open generator, evaluator
            self.workers.run()
            self.server()

        finally:
            self.shutdown()


if __name__ == '__main__':
    with open('config.yaml') as f:
        args = yaml.load(f)
    print(args)

    train_args = args['train_args']
    env_args = args['env_args']
    train_args['env'] = env_args

    gym.prepare(env_args)
    learner = Learner(train_args)
    learner.run()
