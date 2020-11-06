# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Parallel Tic-Tac-Toe

import copy
import random

import numpy as np

from environment import BaseEnvironment


class Environment(BaseEnvironment):
    X, Y = 'ABC',  '123'
    BLACK, WHITE = 1, -1
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self, args=None):
        super().__init__()
        self.reset()

    def reset(self, args=None):
        self.board = np.zeros((3, 3))  # (x, y)
        self.win_color = 0
        self.turn_count = 0

    def action2str(self, a):
        return self.X[a // 3] + self.Y[a % 3]

    def str2action(self, s):
        return self.X.find(s[0]) * 3 + self.Y.find(s[1])

    def __str__(self):
        s = '  ' + ' '.join(self.Y) + '\n'
        for i in range(3):
            s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
        return s

    def plays(self, actions):
        # state transition function
        # action is integer (0 ~ 8) or string (sequence)

        selected_player = random.choice(list(actions.keys()))
        selected_color = [self.BLACK, self.WHITE][selected_player]
        action = actions[selected_player]

        x, y = action // 3, action % 3
        self.board[x, y] = selected_color

        # check winning condition
        if self.board[x, :].sum() == 3 * selected_color \
          or self.board[:, y].sum() == 3 * selected_color \
          or (x == y and np.diag(self.board, k=0).sum() == 3 * selected_color) \
          or (x == 2 - y and np.diag(self.board[::-1, :], k=0).sum() == 3 * selected_color):
            self.win_color = selected_color

        self.turn_count += 1

    def turn(self):
        return NotImplementedError()

    def turns(self):
        return self.players()

    def terminal(self):
        # check whether the state is terminal
        return self.win_color != 0 or self.turn_count == 3 * 3

    def reward(self):
        # terminal reward
        rewards = [0, 0]
        if self.win_color > 0:
            rewards = [1, -1]
        if self.win_color < 0:
            rewards = [-1, 1]
        return {p: rewards[idx] for idx, p in enumerate(self.players())}

    def legal_actions(self, player):
        # legal action list
        return [a for a in range(3 * 3) if self.board[a // 3, a % 3] == 0]

    def action_length(self):
        # maximum size of policy (it determines output size of policy function)
        return 3 * 3

    def players(self):
        return [0, 1]

    def observation(self, player=None):
        # input feature for neural nets
        player = player if player is not None else 0
        color = [self.BLACK, self.WHITE][player]
        a = np.stack([
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
            action_map = {}
            for p in e.turns():
                actions = e.legal_actions(p)
                print([e.action2str(a) for a in actions])
                action_map[p] = random.choice(actions)
            e.plays(action_map)
        print(e)
        print(e.reward())
