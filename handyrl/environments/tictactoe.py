# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Tic-Tac-Toe

import copy
import random

import numpy as np

from ..environment import BaseEnvironment


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
        # action is integer (0 ~ 8) or string (sequence)
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return

        x, y = action // 3, action % 3
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

    def play_info(self, info):
        if info != "":
            self.play(info)

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

    def action_length(self):
        # maximum size of policy (it determines output size of policy function)
        return 3 * 3

    def players(self):
        return [0, 1]

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
        print(e.reward())
