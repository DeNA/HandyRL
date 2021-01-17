# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# OpenAI Gym

import gym
import numpy as np
import torch
import torch.nn as nn

from ..environment import BaseEnvironment
from ..model import BaseModel


class SimpleDenseNet(BaseModel):
    def __init__(self, env, args):
        super().__init__(env)
        input_dim = np.prod(env.observation().shape)
        layers, filters = args.get('layers', 1), args.get('filters', 16)
        self.encoder = nn.Linear(input_dim, filters)
        self.blocks = nn.ModuleList(
            [nn.Linear(filters, filters) for _ in range(layers)])
        self.head_p = nn.Linear(filters, self.action_length)
        self.head_r = nn.Linear(filters, 1)

    def forward(self, x, hidden=None):
        h = x.view(x.size(0), -1)
        h = self.encoder(h)
        for block in self.blocks:
            h = block(h)
        p = self.head_p(h)
        r = self.head_r(h)
        return p, None, r, None


class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.env = gym.make('MountainCar-v0')
        self.reset()

    def step_info(self, infos):
        obs, reward, done, info = infos
        self.latest_obs = obs
        self.latest_reward = reward
        self.done = done
        self.latest_info = info
        self.total_reward += reward

    def reset(self, args={}):
        self.reset_info(self.env.reset())

    def reset_info(self, obs):
        self.total_reward = 0
        return self.step_info((obs, 0, False, {}))

    def play(self, action):
        self.play_info(self.env.step(action))

    def play_info(self, infos):
        return self.step_info(infos)

    def diff_info(self):
        return self.latest_obs, self.latest_reward, self.done, self.info

    def terminal(self):
        return self.done

    def reward(self):
        return {0: self.latest_reward}

    def outcome(self):
        return {0: self.total_reward}

    def legal_actions(self):
        return [0, 1]

    def action_length(self):
        return 2

    def net(self):
        return SimpleDenseNet

    def observation(self, _=None):
        return np.array(self.latest_obs)


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
