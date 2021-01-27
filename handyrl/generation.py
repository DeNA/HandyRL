# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np


class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, models, args):
        # episode generation
        moments = []
        hidden = {}
        for player in self.env.players():
            hidden[player] = models[player].init_hidden()

        err = self.env.reset()
        if err:
            return None

        while not self.env.terminal():
            err = self.env.chance()
            if err:
                return None
            if self.env.terminal():
                break

            moment = {'observation': {}, 'value': {}, 'reward': {}, 'return': {}}

            for player in self.env.players():
                obs, v = None, None
                if player == self.env.turn() or self.args['observation']:
                    obs = self.env.observation(player)
                    model = models[player]
                    outputs = model.inference(obs, hidden[player])
                    hidden[player] = outputs.get('hidden', None)
                    v = outputs.get('value', None)
                    if player == self.env.turn():
                        p = outputs['policy']
                        legal_actions = self.env.legal_actions()
                        action_mask = np.ones_like(p) * 1e32
                        action_mask[legal_actions] = 0
                        p_turn = p - action_mask
                moment['observation'][player] = obs
                moment['value'][player] = v

            def softmax(x):
                x = np.exp(x - np.max(x, axis=-1))
                return x / x.sum(axis=-1)

            action = random.choices(legal_actions, weights=softmax(p_turn[legal_actions]))[0]

            moment['policy'] = p_turn
            moment['action_mask'] = action_mask
            moment['turn'] = self.env.turn()
            moment['action'] = action
            moments.append(moment)

            err = self.env.play(action)
            if err:
                return None

            reward = self.env.reward()
            for player in self.env.players():
                moment['reward'][player] = reward.get(player, None)

        if len(moments) < 1:
            return None

        for player in self.env.players():
            ret = 0
            for i, m in reversed(list(enumerate(moments))):
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret

        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'moment': [
                bz2.compress(pickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ]
        }

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
