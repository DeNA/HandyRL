# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np

from model import softmax
from connection import send_recv


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

            moment = {'observation': {}, 'policy': {}, 'value': {}, 'pmask': {}, 'action': {}}

            turn_players = self.env.turns()
            for index, player in enumerate(self.env.players()):
                obs, p, v, pmask, action = None, None, None, None, None
                if player in turn_players or self.args['observation']:
                    obs = self.env.observation(player)
                    model = models[player]
                    p_, v, hidden[player] = model.inference(obs, hidden[player])
                    if player in turn_players:
                        legal_actions = self.env.legal_actions(player)
                        pmask = np.ones_like(p_) * 1e32
                        pmask[legal_actions] = 0
                        p = p_ - pmask
                        action = random.choices(legal_actions, weights=softmax(p[legal_actions]))[0]

                moment['observation'][index] = obs
                moment['value'][index] = v
                moment['policy'][index] = p
                moment['pmask'][index] = pmask
                moment['action'][index] = action

            moments.append(bz2.compress(pickle.dumps(moment)))

            err = self.env.plays(moment['action'])
            if err:
                return None

        if len(moments) < 1:
            return None

        rewards = self.env.reward()
        rewards = [rewards[player] for player in self.env.players()]

        episode = {'args': args, 'moment': moments, 'reward': rewards}

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
