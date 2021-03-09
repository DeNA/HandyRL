# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# OpenAI Gym

import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..environment import BaseEnvironment
from ..model import BaseModel


class SimpleDenseNet(BaseModel):
    def __init__(self, env, args):
        super().__init__(env)
        input_dim = np.prod(env.observation().shape)
        layers, filters = 2, 16
        self.bn = nn.BatchNorm1d(input_dim)
        self.encoder = nn.Linear(input_dim, filters)
        self.blocks = nn.ModuleList([nn.Linear(filters, filters) for _ in range(layers)])
        self.head_p = nn.Linear(filters, self.action_length, bias=False)
        self.head_r = nn.Linear(filters, 1, bias=False)

    def forward(self, x, hidden=None):
        h = self.bn(x.view(x.size(0), -1))
        h = F.relu(self.encoder(h))
        for block in self.blocks:
            h = F.relu(block(h))
        p = self.head_p(h)
        r = self.head_r(h)
        return {'policy': p, 'return': r}


class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.env = gym.make('MountainCar-v0')
        self.reset()

    def update(self, infos, reset):
        if reset:
            self.obses = []
            self.total_reward = 0
        obs, reward, done, info = copy.deepcopy(infos)
        self.obses.append(obs)
        self.latest_reward = reward
        self.done = done
        self.latest_info = info
        self.total_reward += reward

    def reset(self, args={}):
        self.update((self.env.reset(), 0, False, {}), True)

    def step(self, actions):
        self.update(self.env.step(actions[0]), False)

    def diff_info(self, _=None):
        return self.obses[-1], self.latest_reward, self.done, self.latest_info

    def terminal(self):
        return self.done

    def reward(self):
        return {0: self.latest_reward}

    def outcome(self):
        return {0: self.total_reward / 200}

    def legal_actions(self, _=None):
        return [0, 1]

    def action_length(self):
        return 2

    def net(self):
        return SimpleDenseNet

    def observation(self, _=None):
        history = [self.obses[0]] * (4 - len(self.obses)) + self.obses[-4:]
        return np.array(history, dtype=np.float32)


if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = e.legal_actions()
            print([e.action2str(a) for a in actions])
            e.play(random.choice(actions))
        print(e)
        print(e.reward())
