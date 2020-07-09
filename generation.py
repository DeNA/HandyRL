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
        behaviors, pmasks = [], []
        turns, policies = [], []
        observations, values = {}, {}
        hidden = {}
        for index, player in enumerate(self.env.players()):
            hidden[index] = models[player].init_hidden()
            observations[index] = []
            values[index] = []

        err = self.env.reset()
        if err:
            return None

        while not self.env.terminal():
            err = self.env.chance()
            if err:
                return None
            if self.env.terminal():
                break

            for index, player in enumerate(self.env.players()):
                observation, v = None, None
                if player == self.env.turn() or self.args['observation']:
                    observation = self.env.observation(player)
                    model = models[player]
                    p, v, hidden[index] = model.inference(observation, hidden[index])
                    if player == self.env.turn():
                        legal_actions = self.env.legal_actions()
                        pmask = np.ones_like(p) * 1e32
                        pmask[legal_actions] = 0
                        p_turn = p - pmask
                        index_turn = index
                observations[index].append(observation)
                values[index].append(v)

            action = random.choices(legal_actions, weights=softmax(p_turn[legal_actions]))[0]

            policies.append(p_turn)
            pmasks.append(pmask)
            turns.append(index_turn)
            behaviors.append(action)

            err = self.env.play(action)
            if err:
                return None

        if len(turns) < 1:
            return None
        rewards = self.env.reward()
        rewards = [rewards[player] for player in self.env.players()]

        episode = {
            'args': args, 'observation': observations, 'turn': turns, 'pmask': pmasks,
            'action': behaviors, 'reward': rewards,
            'policy': policies, 'value': values
        }

        return bz2.compress(pickle.dumps(episode))

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
