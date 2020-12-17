# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np

from .model import softmax


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

            moment = {'observation': {}, 'value': {}}

            for index, player in enumerate(self.env.players()):
                obs, v = None, None
                if player == self.env.turn() or self.args['observation']:
                    obs = self.env.observation(player)
                    model = models[player]
                    p, v, hidden[player] = model.inference(obs, hidden[player])
                    if player == self.env.turn():
                        legal_actions = self.env.legal_actions()
                        pmask = np.ones_like(p) * 1e32
                        pmask[legal_actions] = 0
                        p_turn = p - pmask
                        index_turn = index
                moment['observation'][index] = obs
                moment['value'][index] = v

            action = random.choices(legal_actions, weights=softmax(p_turn[legal_actions]))[0]

            moment['policy'] = p_turn
            moment['pmask'] = pmask
            moment['turn'] = index_turn
            moment['action'] = action
            moments.append(moment)

            err = self.env.play(action)
            if err:
                return None

        if len(moments) < 1:
            return None

        rewards = self.env.reward()
        rewards = [rewards[player] for player in self.env.players()]

        episode = {
            'args': args, 'steps': len(moments), 'reward': rewards,
            'moment': [
                bz2.compress(pickle.dumps(moments[i:i+self.args['compress_steps']])) \
                    for i in range(0, len(moments), self.args['compress_steps'])
            ],
        }

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
