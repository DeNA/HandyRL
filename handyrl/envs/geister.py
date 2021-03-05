# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Geister

import random
import itertools

import numpy as np
import torch
import torch.nn as nn

from ..environment import BaseEnvironment
from ..model import BaseModel, Encoder, Head, DRC, Conv


class GeisterNet(BaseModel):
    def __init__(self, env, args={}):
        super().__init__(env, args)

        layers, filters, p_filters = 3, 32, 8
        o = env.observation()
        input_channels = o['scalar'].shape[-1] + o['board'].shape[-3]
        self.input_size = (input_channels, 6, 6)

        self.encoder = Encoder(self.input_size, filters)
        self.body = DRC(layers, filters, filters)
        self.head_p1 = Conv(filters * 2, p_filters, 1, bn=False)
        self.activation_p = nn.LeakyReLU(0.1)
        self.head_p2 = Conv(p_filters, 4, 1, bn=False, bias=False)
        self.head_p_set = nn.Linear(1, 70 * 2, bias=True)
        self.head_v = Head((filters * 2, 6, 6), 1, 1)
        self.head_r = Head((filters * 2, 6, 6), 1, 1)

    def init_hidden(self, batch_size=None):
        return self.body.init_hidden(self.input_size[1:], batch_size)

    def forward(self, x, hidden):
        b, s = x['board'], x['scalar']
        h_s = s.view(*s.size(), 1, 1).repeat(1, 1, 6, 6)
        h = torch.cat([h_s, b], -3)

        h_e = self.encoder(h)
        h, hidden = self.body(h_e, hidden, num_repeats=3)

        h = torch.cat([h_e, h], -3)
        h_p = self.activation_p(self.head_p1(h))
        h_p = self.head_p2(h_p).view(*h.size()[:-3], 4 * 6 * 6)
        h_p_set = self.head_p_set(s.sum(-1, keepdim=True).mul(0))
        h_p = torch.cat([h_p, h_p_set], -1)
        h_v = self.head_v(h)
        h_r = self.head_r(h)

        return {'policy': h_p, 'value': torch.tanh(h_v), 'return': h_r, 'hidden': hidden}


class Environment(BaseEnvironment):
    X, Y = 'ABCDEF', '123456'
    BLACK, WHITE = 0, 1
    BLUE, RED = 0, 1
    C = 'BW'
    T = 'BR'
    P = {-1: '_', 0: 'B', 1: 'R', 2: 'b', 3: 'r'}
    # original positions to set pieces
    OPOS = [
        ['B2', 'C2', 'D2', 'E2', 'B1', 'C1', 'D1', 'E1'],
        ['E5', 'D5', 'C5', 'B5', 'E6', 'D6', 'C6', 'B6'],
    ]
    # goal positions
    GPOS = np.array([
        [(-1, 5), (6, 5)],
        [(-1, 0), (6, 0)]
    ], dtype=np.int32)

    D = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)], dtype=np.int32)
    OSEQ = list(itertools.combinations([i for i in range(8)], 4))

    def __init__(self, args=None):
        super().__init__()
        self.reset()

    def reset(self, args={}):
        self.args = args
        self.board = -np.ones((6, 6), dtype=np.int32)  # (x, y) -1 is empty
        self.color = self.BLACK
        self.turn_count = -2  # before setting original positions
        self.win_color = None
        self.piece_cnt = np.zeros(4, dtype=np.int32)
        self.board_index = -np.ones((6, 6), dtype=np.int32)
        self.piece_position = np.zeros((2 * 8, 2), dtype=np.int32)
        self.record = []
        self.captured_type = None
        self.layouts = {}

    def put_piece(self, piece, pos, piece_idx):
        self.board[pos[0], pos[1]] = piece
        self.piece_position[piece_idx] = pos
        self.board_index[pos[0], pos[1]] = piece_idx
        self.piece_cnt[piece] += 1

    def remove_piece(self, piece, pos):
        self.board[pos[0], pos[1]] = -1
        piece_idx = self.board_index[pos[0], pos[1]]
        self.board_index[pos[0], pos[1]] = -1
        self.piece_position[piece_idx] = np.array((-1, -1))
        self.piece_cnt[piece] -= 1

    def move_piece(self, piece, pos_from, pos_to):
        self.board[pos_from[0], pos_from[1]] = -1
        self.board[pos_to[0], pos_to[1]] = piece
        piece_idx = self.board_index[pos_from[0], pos_from[1]]
        self.board_index[pos_from[0], pos_from[1]] = -1
        self.board_index[pos_to[0], pos_to[1]] = piece_idx
        self.piece_position[piece_idx] = pos_to

    def set_pieces(self, c, seq_idx):
        # decide original positions
        chosen_seq = self.OSEQ[seq_idx]
        for idx in range(8):
            t = 0 if idx in chosen_seq else 1
            piece = self.colortype2piece(c, t)
            pos = self.str2position(self.OPOS[c][idx])
            self.put_piece(piece, pos, c * 8 + idx)

    def opponent(self, color):
        return self.BLACK + self.WHITE - color

    def onboard(self, pos):
        return 0 <= pos[0] and pos[0] < 6 and 0 <= pos[1] and pos[1] < 6

    def goal(self, c, pos):
        # check whether pos is goal position for c
        for g in self.GPOS[c]:
            if g[0] == pos[0] and g[1] == pos[1]:
                return True
        return False

    def colortype2piece(self, c, t):
        return c * 2 + t

    def piece2color(self, p):
        return -1 if p == -1 else p // 2

    def piece2type(self, p):
        return -1 if p == -1 else p % 2

    def rotate(self, pos):
        return np.array((5 - pos[0], 5 - pos[1]), dtype=np.int32)

    def position2str(self, pos):
        if self.onboard(pos):
            return self.X[pos[0]] + self.Y[pos[1]]
        else:
            return '**'

    def str2position(self, s):
        if s != '**':
            return np.array((self.X.find(s[0]), self.Y.find(s[1])), dtype=np.int32)
        else:
            return None

    def fromdirection2action(self, pos_from, d, c):
        if c == self.WHITE:
            pos_from = self.rotate(pos_from)
            d = 3 - d
        return d * 36 + pos_from[0] * 6 + pos_from[1]

    def action2from(self, a, c):
        pos1d = a % 36
        pos = np.array((pos1d / 6, pos1d % 6), dtype=np.int32)
        if c == self.WHITE:
            pos = self.rotate(pos)
        return pos

    def action2direction(self, a, c):
        d = a // 36
        if c == self.WHITE:
            d = 3 - d
        return d

    def action2to(self, a, c):
        return self.action2from(a, c) + self.D[self.action2direction(a, c)]

    def action2str(self, a, player):
        if a >= 4 * 6 * 6:
            return 's' + str((a - 4 * 6 * 6) % 70)

        c = player
        pos_from = self.action2from(a, c)
        pos_to = self.action2to(a, c)
        return self.position2str(pos_from) + self.position2str(pos_to)

    def str2action(self, s, player):
        if s[0] == 's':
            return 4 * 6 * 6 + 70 * player + int(s[1:])

        c = player
        pos_from = self.str2position(s[:2])
        pos_to = self.str2position(s[2:])

        if pos_to is None:
            # it should arrive at a goal
            for g in self.GPOS[c]:
                if ((pos_from - g) ** 2).sum() == 1:
                    diff = g - pos_from
                    for d, dd in enumerate(self.D):
                        if np.array_equal(dd, diff):
                            break
                    break
        else:
            # check action direction
            diff = pos_to - pos_from
            for d, dd in enumerate(self.D):
                if np.array_equal(dd, diff):
                    break

        return self.fromdirection2action(pos_from, d, c)

    def record_string(self):
        return ' '.join([self.action2str(a, i % 2) for i, a in enumerate(self.record)])

    def position_string(self):
        poss = [self.position2str(pos) for pos in self.piece_position]
        return ','.join(poss)

    def __str__(self):
        # output state
        s = '  ' + ' '.join(self.Y) + '\n'
        for i in range(6):
            s += self.X[i] + ' ' + ' '.join([self.P[self.board[i, j]] for j in range(6)]) + '\n'
        s += 'color = ' + self.C[self.color] + '\n'
        s += 'record = ' + self.record_string()
        return s

    def _set(self, layout):
        if layout < 0:
            layout = random.randrange(70)
        self.layouts[self.color] = layout
        self.set_pieces(self.color, layout)
        self.color = self.opponent(self.color)
        self.turn_count += 1

    def play(self, action, _=None):
        # state transition
        if self.turn_count < 0:
            layout = action - 4 * 6 * 6 - 70 * self.color
            return self._set(layout)

        ox, oy = self.action2from(action, self.color)
        nx, ny = self.action2to(action, self.color)
        piece = self.board[ox, oy]
        self.captured_type = None

        if not self.onboard((nx, ny)):
            # finish by goal
            self.remove_piece(piece, (ox, oy))
            self.win_color = self.color
        else:
            piece_cap = self.board[nx, ny]
            if piece_cap != -1:
                # capture opponent piece
                self.remove_piece(piece_cap, (nx, ny))
                if self.piece_cnt[piece_cap] == 0:
                    if self.piece2type(piece_cap) == self.BLUE:
                        # win by capturing all opponent blue pieces
                        self.win_color = self.color
                    else:
                        # lose by capturing all opponent red pieces
                        self.win_color = self.opponent(self.color)
                self.captured_type = self.piece2type(piece_cap)

            # move piece
            self.move_piece(piece, (ox, oy), (nx, ny))

        self.color = self.opponent(self.color)
        self.turn_count += 1
        self.record.append(action)

        if self.turn_count >= 200 and self.win_color is None:
            self.win_color = 2  # draw

    def diff_info(self, player):
        color = player
        played_color = (self.turn_count - 1) % 2
        info = {}
        if len(self.record) == 0:
            if self.turn_count > -2:
                info['set'] = self.layouts[played_color] if color == played_color else -1
        else:
            info['move'] = self.action2str(self.record[-1], played_color)
            if color == played_color and self.captured_type is not None:
                info['captured'] = self.T[self.captured_type]
        return info

    def update(self, info, reset):
        if reset:
            self.args = {**self.args, **info}
            self.reset(info)
        elif 'set' in info:
            self._set(info['set'])
        elif 'move' in info:
            action = self.str2action(info['move'], self.color)
            if 'captured' in info:
                # set color to captured piece
                pos_to = self.action2to(action, self.color)
                t = self.T.index(info['captured'])
                piece = self.colortype2piece(self.opponent(self.color), t)
                self.board[pos_to[0], pos_to[1]] = piece
            self.play(action)

    def turn(self):
        return self.players()[self.turn_count % 2]

    def terminal(self):
        # check whether terminal state or not
        return self.win_color is not None

    def reward(self):
        # return immediate rewards
        return {p: -0.01 for p in self.players()}

    def outcome(self):
        # return terminal outcomes
        outcomes = [0, 0]
        if self.win_color == self.BLACK:
            outcomes = [1, -1]
        elif self.win_color == self.WHITE:
            outcomes = [-1, 1]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def legal(self, action):
        if self.turn_count < 0:
            layout = action - 4 * 6 * 6 - 70 * self.color
            return 0 <= layout < 70

        pos_from = self.action2from(action, self.color)
        pos_to = self.action2to(action, self.color)

        piece = self.board[pos_from[0], pos_from[1]]
        c, t = self.piece2color(piece), self.piece2type(piece)
        if c != self.color:
            # no piece on destination position
            return False

        return self._legal(c, t, pos_from, pos_to)

    def _legal(self, c, t, pos_from, pos_to):
        if self.onboard(pos_to):
            # can move to cell if there isn't my piece
            piece_cap = self.board[pos_to[0], pos_to[1]]
            return self.piece2color(piece_cap) != c
        else:
            # can move to my goal
            return t == self.BLUE and self.goal(c, pos_to)

    def legal_actions(self, _=None):
        # return legal action list
        if self.turn_count < 0:
            return [4 * 6 * 6 + 70 * self.color + i for i in range(70)]
        actions = []
        for pos in self.piece_position[self.color*8:(self.color+1)*8]:
            if pos[0] == -1:
                continue
            t = self.piece2type(self.board[pos[0], pos[1]])
            for d in range(4):
                if self._legal(self.color, t, pos, pos + self.D[d]):
                    action = self.fromdirection2action(pos, d, self.color)
                    actions.append(action)

        return actions

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return 4 * 6 * 6 + 70 * 2

    def players(self):
        return [0, 1]

    def observation(self, player=None):
        # state representation to be fed into neural networks
        turn_view = player is None or player == self.turn()
        color = self.color if turn_view else self.opponent(self.color)
        opponent = self.opponent(color)

        nbcolor = self.piece_cnt[self.colortype2piece(color,    self.BLUE)]
        nrcolor = self.piece_cnt[self.colortype2piece(color,    self.RED )]
        nbopp   = self.piece_cnt[self.colortype2piece(opponent, self.BLUE)]
        nropp   = self.piece_cnt[self.colortype2piece(opponent, self.RED )]

        s = np.array([
            1 if turn_view           else 0,  # view point is turn player
            1 if color == self.BLACK else 0,  # my color is black
            # the number of remained pieces
            *[(1 if nbcolor == i else 0) for i in range(1, 5)],
            *[(1 if nrcolor == i else 0) for i in range(1, 5)],
            *[(1 if nbopp   == i else 0) for i in range(1, 5)],
            *[(1 if nropp   == i else 0) for i in range(1, 5)]
        ]).astype(np.float32)

        blue_c = self.board == self.colortype2piece(color,    self.BLUE)
        red_c  = self.board == self.colortype2piece(color,    self.RED)
        blue_o = self.board == self.colortype2piece(opponent, self.BLUE)
        red_o  = self.board == self.colortype2piece(opponent, self.RED)

        b = np.stack([
            # board zone
            np.ones_like(self.board),
            # my/opponent's all pieces
            blue_c + red_c,
            blue_o + red_o,
            # my blue/red pieces
            blue_c,
            red_c,
            # opponent's blue/red pieces
            blue_o if player is None else np.zeros_like(self.board),
            red_o  if player is None else np.zeros_like(self.board)
        ]).astype(np.float32)

        if color == self.WHITE:
            b = np.rot90(b, k=2, axes=(1, 2))

        return {'scalar': s, 'board': b}

    def net(self):
        return GeisterNet


if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = e.legal_actions()
            print([e.action2str(a, e.turn()) for a in actions])
            e.play(random.choice(actions))
        print(e)
        print(e.outcome())
