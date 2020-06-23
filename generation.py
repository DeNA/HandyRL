# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np

from model import softmax, ModelCongress
from connection import send_recv


class Generator:
    def __init__(self, env, args, conn):
        self.env = env
        self.args = args
        self.conn = conn
        self.latest_model = -1, None  # id, model

    def generate(self, models, args):
        # episode generation
        behaviors, pmasks = [], []
        turns, policies = [], []
        observations, values = {}, {}
        hidden = {}
        for player, model in models.items():
            hidden[player] = model.init_hidden()
            observations[player] = []
            values[player] = []

        err = self.env.reset()
        if err:
            return None

        while not self.env.terminal():
            err = self.env.chance()
            if err:
                return None
            if self.env.terminal():
                break

            for player, model in models.items():
                observation, v = None, None
                if player == self.env.turn() or self.args['observation']:
                    observation = self.env.observation(player)
                    p, v, hidden[player] = model.inference(observation, hidden[player])
                    if player == self.env.turn():
                        legal_actions = self.env.legal_actions()
                        pmask = np.ones_like(p) * 1e32
                        pmask[legal_actions] = 0
                        p_turn = p - pmask
                observations[player].append(observation)
                values[player].append(v)

            action = random.choices(legal_actions, weights=softmax(p_turn[legal_actions]))[0]

            policies.append(p_turn)
            pmasks.append(pmask)
            turns.append(self.env.turn())
            behaviors.append(action)

            err = self.env.play(action)
            if err:
                return None

        if len(turns) < 1:
            return None

        episode = {
            'args': args, 'observation': observations, 'turn': turns, 'pmask': pmasks,
            'action': behaviors, 'reward': self.env.reward(),
            'policy': policies, 'value': values
        }

        return bz2.compress(pickle.dumps(episode))

    def _gather_models(self, model_ids):
        model_pool = {}
        for model_id in model_ids:
            if model_id not in model_pool:
                if model_id == self.latest_model[0]:
                    # use latest model
                    model_pool[model_id] = self.latest_model[1]
                else:
                    # get model from server
                    model_pool[model_id] = send_recv(self.conn, ('model', model_id))
                    # update latest model
                    if model_id > self.latest_model[0]:
                        self.latest_model = model_id, model_pool[model_id]
        return model_pool

    def execute(self):
        args = send_recv(self.conn, ('gargs', None))
        model_pool = self._gather_models([idx for m in args['model_id'].values() for idx in m])

        # make dict of models
        models = {}
        for p, model_ids in args['model_id'].items():
            if len(model_ids) == 1:
                models[p] = model_pool[model_ids[0]]
            else:
                models[p] = ModelCongress([model_pool[model_id] for model_id in model_ids])

        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')

        continue_flag = send_recv(self.conn, ('episode', episode))
        return continue_flag
