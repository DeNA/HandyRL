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
    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player)
        else:
            return random.choice(env.legal_actions(player))


def print_outputs(env, action, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(action, prob, v)
    else:
        print('v = %f' % v)
        print('a = %d prob = %f' % (action, prob))


class Agent:
    def __init__(self, model, observation=False, temperature=1e-6):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.observation = observation
        self.temperature = temperature

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs, legal_actions):
        outputs = self.model.inference(obs, self.hidden, legal_actions=legal_actions, temperature=self.temperature)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        obs = env.observation(player)
        legal_actions = env.legal_actions(player)
        outputs = self.plan(obs, legal_actions)

        action = outputs['action']
        prob = outputs['selected_prob']
        v = outputs.get('value', None)

        if show:
            print_outputs(env, action, selected_prob, v)

        return action

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            outputs = self.plan(env.observation(player))
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v if v is not None else [0.0]


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o:
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [o]
        for k, vl in outputs:
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model, observation=False):
        super().__init__(model, observation=observation, temperature=1.0)
