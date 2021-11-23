# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Tic-Tac-Toe

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..environment import BaseEnvironment
from ..search import MonteCarloTree
from ..model import to_torch


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(
            filters0, filters1, kernel_size,
            stride=1, padding=kernel_size//2, bias=bias
        )
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class Head(nn.Module):
    def __init__(self, input_size, out_filters, outputs):
        super().__init__()

        self.board_size = input_size[1] * input_size[2]
        self.out_filters = out_filters

        self.conv = Conv(input_size[0], out_filters, 1, bn=False)
        self.activation = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(self.board_size * out_filters, outputs, bias=False)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = self.fc(h.view(-1, self.board_size * self.out_filters))
        return h


class SimpleConv2dModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 3, 32

        self.conv = nn.Conv2d(3, filters, 3, stride=1, padding=1)
        self.blocks = nn.ModuleList([Conv(filters, filters, 3, bn=True) for _ in range(layers)])
        self.head_p = Head((filters, 3, 3), 2, 9)
        self.head_v = Head((filters, 3, 3), 1, 1)

    def forward(self, x, hidden=None):
        h = F.relu(self.conv(x))
        for block in self.blocks:
            h = F.relu(block(h))
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        return {'policy': h_p, 'value': torch.tanh(h_v)}


class ResidualBlock(nn.Module):
    def __init__(self, filters0, filters1):
        super().__init__()
        self.conv = Conv(filters0, filters1, 3, bn=True)

    def forward(self, x):
        h = self.conv(x)
        return F.relu_(x + h)


class MuZero(nn.Module):
    class Representation(nn.Module):
        ''' Conversion from observation to inner abstract state '''
        def __init__(self, input_dim, layers, filters):
            super().__init__()
            self.layer0 = Conv(input_dim, filters, 3, bn=True)
            self.blocks = nn.ModuleList([ResidualBlock(filters, filters) for _ in range(layers)])

        def forward(self, x):
            h = F.relu_(self.layer0(x))
            for block in self.blocks:
                h = block(h)
            return h

        def inference(self, x):
            self.eval()
            with torch.no_grad():
                rp = self(to_torch(x).unsqueeze(0))
            return rp.cpu().numpy().squeeze(0)

    class Prediction(nn.Module):
        ''' Policy and value prediction from inner abstract state '''
        def __init__(self, internal_size, num_players, action_length):
            super().__init__()
            self.head_p = Head(internal_size, 4, num_players * action_length)
            self.head_v = Head(internal_size, 2, num_players)

        def forward(self, rp):
            p = self.head_p(rp)
            v = self.head_v(rp)
            return p, torch.tanh(v)

        def inference(self, rp):
            self.eval()
            with torch.no_grad():
                p, v = self(to_torch(rp).unsqueeze(0))
            return p.cpu().numpy().squeeze(0), v.cpu().numpy().squeeze(0)

    class Dynamics(nn.Module):
        '''Abstract state transition'''
        def __init__(self, rp_shape, layers, num_players, action_length, action_filters):
            super().__init__()
            self.action_shape = action_filters, rp_shape[1], rp_shape[2]
            filters = rp_shape[0]
            self.action_embedding = nn.Embedding(num_players * action_length, embedding_dim=np.prod(self.action_shape))
            self.layer0 = Conv(filters + action_filters, filters, 3, bn=True)
            self.blocks = nn.ModuleList([ResidualBlock(filters, filters) for _ in range(layers)])

        def forward(self, rp, a):
            arp = self.action_embedding(a).view(-1, *self.action_shape)
            h = torch.cat([rp, arp], dim=1)
            h = F.relu_(self.layer0(h))
            for block in self.blocks:
                h = block(h)
            return h

        def inference(self, rp, a):
            self.eval()
            with torch.no_grad():
                rp = self(to_torch(rp).unsqueeze(0), to_torch(a).unsqueeze(0))
            return rp.cpu().numpy().squeeze(0)

    def __init__(self, env, obs, action_length):
        super().__init__()
        self.num_players = len(env.players())
        self.action_length = action_length
        self.input_size = obs.shape

        layers, filters = 3, 32
        action_filters = 4
        internal_size = (filters, *self.input_size[1:])
        self.planning_args = {
            'root_noise_alpha': 0.15,
            'root_noise_coef': 0.25,
        }

        self.nets = nn.ModuleDict({
            'representation': self.Representation(self.input_size[0], layers, filters),
            'prediction': self.Prediction(internal_size, self.num_players, self.action_length),
            'dynamics': self.Dynamics(internal_size, layers, self.num_players, self.action_length, action_filters),
        })

    def init_hidden(self, batch_size=None):
        return {}

    def forward(self, x, hidden, action=None):
        if 'representation' not in hidden:
            rp = self.nets['representation'](x)
        else:
            rp = hidden['representation']
        p, v = self.nets['prediction'](rp)
        outputs = {'policy': p, 'value': v}

        if action is not None:
            next_rp = self.nets['dynamics'](rp, action)
            outputs['hidden'] = {'representation': next_rp}
        return outputs

    def inference(self, x, hidden=None, num_simulations=30):
        tree = MonteCarloTree(self.nets, self.planning_args)
        p, v = tree.think(x, num_simulations)
        return {'policy': p, 'value': v}



class Environment(BaseEnvironment):
    X, Y = 'ABC',  '123'
    BLACK, WHITE = 1, -1
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self, args=None):
        super().__init__()
        self.reset()

    def reset(self, args=None):
        self.board = np.zeros((3, 3))  # (x, y)
        self.color = self.BLACK
        self.win_color = 0
        self.record = []

    def action2str(self, a, _=None):
        pos = a % 9
        return self.X[pos // 3] + self.Y[pos % 3]

    def str2action(self, s, player):
        pos = self.X.find(s[0]) * 3 + self.Y.find(s[1])
        return pos + 9 * player

    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        s = '  ' + ' '.join(self.Y) + '\n'
        for i in range(3):
            s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
        s += 'record = ' + self.record_string()
        return s

    def play(self, action, _=None):
        # state transition function
        # action is integer (0 ~ 8)
        pos = action % 9
        x, y = pos // 3, pos % 3
        self.board[x, y] = self.color

        # check winning condition
        win = self.board[x, :].sum() == 3 * self.color \
            or self.board[:, y].sum() == 3 * self.color \
            or (x == y and np.diag(self.board, k=0).sum() == 3 * self.color) \
            or (x == 2 - y and np.diag(self.board[::-1, :], k=0).sum() == 3 * self.color)

        if win:
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)

    def diff_info(self, _):
        if len(self.record) == 0:
            return ""
        return self.action2str(self.record[-1])

    def update(self, info, reset):
        if reset:
            self.reset()
        else:
            action = self.str2action(info, self.turn())
            self.play(action)

    def turn(self):
        return self.players()[len(self.record) % 2]

    def terminal(self):
        # check whether the state is terminal
        return self.win_color != 0 or len(self.record) == 3 * 3

    def outcome(self):
        # terminal outcome
        outcomes = [0, 0]
        if self.win_color > 0:
            outcomes = [1, -1]
        if self.win_color < 0:
            outcomes = [-1, 1]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def legal_actions(self, _=None):
        # legal action list
        player = self.turn()
        return [pos + 9 * player for pos in range(3 * 3) if self.board[pos // 3, pos % 3] == 0]

    def action_length(self):
        # maximum size of policy (it determines output size of policy function)
        return 3 * 3

    def players(self):
        return [0, 1]

    def net(self):
        obs = self.observation(self.players()[0])
        return MuZero(self, obs, 9)

    def observation(self, player=None):
        # input feature for neural nets
        a = np.stack([
            np.ones_like(self.board) if self.turn() == 0 else np.zeros_like(self.board),
            self.board == self.BLACK,
            self.board == self.WHITE,
        ]).astype(np.float32)
        return a


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
        print(e.outcome())
