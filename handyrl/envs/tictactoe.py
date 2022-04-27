# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Tic-Tac-Toe

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from ..environment import BaseEnvironment


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

    def forward(self, x, hidden=None, action=None, temperature=1.0):
        h = F.relu(self.conv(x))
        for block in self.blocks:
            h = F.relu(block(h))
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        log_prob = F.log_softmax(h_p / temperature, -1)
        entropy = dist.Categorical(logits=log_prob).entropy().unsqueeze(-1)

        if action is None:
            prob = torch.exp(log_prob)
            action = prob.multinomial(num_samples=1, replacement=True)
        log_selected_prob = log_prob.gather(-1, action)

        return {'action': action, 'log_selected_prob': log_selected_prob, 'value': torch.tanh(h_v), 'entropy': entropy}


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
        return self.X[a // 3] + self.Y[a % 3]

    def str2action(self, s, _=None):
        return self.X.find(s[0]) * 3 + self.Y.find(s[1])

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
        x, y = action // 3, action % 3
        if self.board[x, y] != 0:  # illegal action
            self.win_color = -self.color
        else:
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
            action = self.str2action(info)
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
        return [a for a in range(3 * 3) if self.board[a // 3, a % 3] == 0]

    def players(self):
        return [0, 1]

    def net(self):
        return SimpleConv2dModel()

    def observation(self, player=None):
        # input feature for neural nets
        turn_view = player is None or player == self.turn()
        color = self.color if turn_view else -self.color
        a = np.stack([
            np.ones_like(self.board) if turn_view else np.zeros_like(self.board),
            self.board == color,
            self.board == -color
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
