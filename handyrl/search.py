# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# tree search

import copy
import time

import numpy as np


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1))
    return x / x.sum(axis=-1)


class Node:
    '''Search result of one abstract (or root) state'''
    def __init__(self, p, v):
        self.p, self.v = p, v
        self.n = np.zeros_like(p)
        self.q_sum = np.zeros((*p.shape, v.shape[-1]), dtype=np.float32)
        self.n_all, self.q_sum_all = 1, v / 2  # prior

    def update(self, action, q_new):
        # Update
        self.n[action] += 1
        self.q_sum[action] += q_new

        # Update overall stats
        self.n_all += 1
        self.q_sum_all += q_new


class MonteCarloTree:
    '''Monte Carlo Tree Search'''
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.nodes = {}

    def search(self, rp, path):
        # Return predicted value from new state
        key = '|' + ' '.join(map(str, path))
        if key not in self.nodes:
            p, v = self.model['prediction'].inference(rp)
            p, v = softmax(p), v
            self.nodes[key] = Node(p, v)
            return v

        # Choose action with bandit
        node = self.nodes[key]
        p = node.p
        if len(path) == 0:
            # Add noise to policy on the root node
            noise = np.random.dirichlet([self.args['root_noise_alpha']] * np.prod(p.shape)).reshape(*p.shape)
            p = (1 - self.args['root_noise_coef']) * p + self.args['root_noise_coef'] * noise
            # On the root node, we choose action only from legal actions
            p /= p.sum() + 1e-16

        q_mean_all = node.q_sum_all.reshape(1, -1) / node.n_all
        n, q_sum = 1 + node.n, q_mean_all + node.q_sum
        adv = (q_sum / n.reshape(-1, 1) - q_mean_all).reshape(q_sum.shape[-1], -1, q_sum.shape[-1])
        adv = np.concatenate([adv[0, :, 0], adv[1, :, 1]])
        ucb = adv + 2.0 * np.sqrt(node.n_all) * p / n  # PUCB formula
        selected_action = np.argmax(ucb)

        # Search next state by recursively calling this function
        next_rp = self.model['dynamics'].inference(rp, np.array([selected_action]))
        path.append(selected_action)
        q_new = self.search(next_rp, path)
        node.update(selected_action, q_new)

        return q_new

    def think(self, root_obs, num_simulations, env=None, show=False):
        # End point of MCTS
        start, prev_time = time.time(), 0
        for _ in range(num_simulations):
            self.search(self.model['representation'].inference(root_obs), [])

            # Display search result on every second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root, pv = self.nodes['|'], self.pv(env)
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s'
                          % (tmp_time, env.action2str(pv[0][0], pv[0][1]), root.q_sum[pv[0][0]] / root.n[pv[0][0]],
                             root.n[pv[0][0]], root.n_all, ' '.join([env.action2str(a, p) for a, p in pv])))

        #  Return probability distribution weighted by the number of simulations
        root = self.nodes['|']
        n = root.n + 0.1
        p = np.log(n / n.sum())
        v = (root.q_sum * p.reshape(-1, 1)).sum(0)
        return p, v

    def pv(self, env_):
        # Return principal variation (action sequence which is considered as the best)
        env = copy.deepcopy(env_)
        pv_seq = []
        while True:
            path = list(zip(*pv_seq))[0]
            key = '|' + ' '.join(map(str, path))
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted([(a, self.nodes[key].n[a]) for a in env.legal_actions()], key=lambda x: -x[1])[0][0]
            pv_seq.append((best_action, env.turn()))
            env.play(best_action)
        return pv_seq
