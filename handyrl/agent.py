# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random

import numpy as np

from .util import softmax


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]


class RuleBasedAgent(RandomAgent):
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player, key=self.key)
        else:
            return random.choice(env.legal_actions(player))


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        if v is not None:
            print('v = %f' % v)
        if prob is not None:
            print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, temperature=0.0, observation=True):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.temperature = temperature
        self.observation = observation

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        obs = env.observation(player)
        outputs = self.plan(obs)
        actions = env.legal_actions(player)
        p = outputs['policy']
        v = outputs.get('value', None)
        mask = np.ones_like(p)
        mask[actions] = 0
        p = p - mask * 1e32

        if show:
            print_outputs(env, softmax(p), v)

        if self.temperature == 0:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            return ap_list[0][0]
        else:
            return random.choices(np.arange(len(p)), weights=softmax(p / self.temperature))[0]

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            obs = env.observation(player)
            outputs = self.plan(obs)
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o.items():
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [v]
        for k, vl in outputs.items():
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model):
        super().__init__(model, temperature=1.0)
