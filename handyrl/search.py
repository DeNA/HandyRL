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
        self.n, self.q_sum = np.zeros_like(p), np.zeros_like(p)
        self.n_all, self.q_sum_all = 1, v / 2 # prior

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
        self.nodes = {}
        self.args = {}

    def search(self, rp, path):
        # Return predicted value from new state
        key = '|' + ' '.join(map(str, path))
        if key not in self.nodes:
            p, v = self.model['prediction'].inference(rp)
            p, v = softmax(p), v[0]
            self.nodes[key] = Node(p, v)
            return v

        # State transition by an action selected from bandit
        node = self.nodes[key]
        p = node.p
        if len(path) == 0:
            # Add noise to policy on the root node
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.15] * len(p))
            # On the root node, we choose action only from legal actions
            p /= p.sum() + 1e-16

        n, q_sum = 1 + node.n, node.q_sum_all / node.n_all + node.q_sum
        ucb = q_sum / n + 2.0 * np.sqrt(node.n_all) * p / n  # PUCB formula
        best_action = np.argmax(ucb)

        # Search next state by recursively calling this function
        next_rp = self.model['dynamics'].inference(rp, np.array([best_action]))
        path.append(best_action)
        q_new = -self.search(next_rp, path)  # With the assumption of changing player by turn
        node.update(best_action, q_new)

        return q_new

    def think(self, root_obs, num_simulations, temperature=0, env=None, show=False):
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
                          % (tmp_time, env.action2str(pv[0]), root.q_sum[pv[0]] / root.n[pv[0]],
                             root.n[pv[0]], root.n_all, ' '.join([env.action2str(a) for a in pv])))

        #  Return probability distribution weighted by the number of simulations
        root = self.nodes['|']
        n = root.n + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        p = n / n.sum()
        v = root.q_sum_all / root.n_all
        return p, v

    def pv(self, env):
        # Return principal variation (action sequence which is considered as the best)
        s, pv_seq = copy.deepcopy(env), []
        while True:
            key = '|'
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted([(a, self.nodes[key].n[a]) for a in env.legal_actions()], key=lambda x: -x[1])[0][0]
            pv_seq.append(best_action)
            s.play(best_action)
        return pv_seq