# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Parallel Tic-Tac-Toe

import random

import numpy as np

from .tictactoe import Environment as TicTacToe


class Environment(TicTacToe):
    def __str__(self):
        s = '  ' + ' '.join(self.Y) + '\n'
        for i in range(3):
            s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
        return s

    def step(self, actions):
        # state transition function
        selected_player = random.choice(list(actions.keys()))
        action = actions[selected_player]
        self._step(action, selected_player)

    def _step(self, action, selected_player):
        selected_color = [self.BLACK, self.WHITE][selected_player]
        x, y = action // 3, action % 3
        self.board[x, y] = selected_color

        # check winning condition
        if self.board[x, :].sum() == 3 * selected_color \
          or self.board[:, y].sum() == 3 * selected_color \
          or (x == y and np.diag(self.board, k=0).sum() == 3 * selected_color) \
          or (x == 2 - y and np.diag(self.board[::-1, :], k=0).sum() == 3 * selected_color):
            self.win_color = selected_color

        self.record.append((selected_color, action))

    def diff_info(self, _):
        if len(self.record) == 0:
            return ""
        color, action = self.record[-1]
        return self.action2str(action) + ":" + self.C[color]

    def update(self, info, reset):
        if reset:
            self.reset()
        else:
            saction, scolor = info.split(":")
            action, player = self.str2action(saction), 'OX'.index(scolor)
            self._step(action, player)

    def turn(self):
        return NotImplementedError()

    def turns(self):
        return self.players()


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
            e.step(action_map)
        print(e)
        print(e.outcome())
