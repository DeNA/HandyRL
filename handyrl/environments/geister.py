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
        self.head_v = Head((filters * 2, 6, 6), 1, 1)

    def init_hidden(self, batch_size=None):
        return self.body.init_hidden(self.input_size[1:], batch_size)

    def forward(self, x, hidden):
        s = x['scalar'].reshape(*x['scalar'].size(), 1, 1).repeat(1, 1, 6, 6)
        h = torch.cat([s, x['board']], -3)

        h_e = self.encoder(h)
        h, hidden = self.body(h_e, hidden, num_repeats=3)

        h = torch.cat([h_e, h], -3)
        h_p = self.activation_p(self.head_p1(h))
        h_p = self.head_p2(h_p).view(*h.size()[:-3], 4 * 6 * 6)
        h_v = self.head_v(h)

        return h_p, torch.tanh(h_v), hidden


class Environment(BaseEnvironment):
    X, Y = 'ABCDEF', '123456'
    BLACK, WHITE = 0, 1
    BLUE, RED = 0, 1
    C = 'BW'
    P = {-1: '_', 0: 'B', 1: 'R', 2: 'b', 3: 'r'}
    _P = {'_': -1, 'B': 0, 'R': 1, 'b': 2, 'r': 3}
    # original positions to set pieces
    OPOS = [
        ['B2', 'C2', 'D2', 'E2', 'B1', 'C1', 'D1', 'E1'],
        ['E5', 'D5', 'C5', 'B5', 'E6', 'D6', 'C6', 'B6'],
    ]
    # goal positions
    GPOS = [['A6', 'F6'], ['A1', 'F1']]

    D = [-6, -1, 1, 6]
    OSEQ = list(itertools.combinations([i for i in range(8)], 4))

    def __init__(self, args=None):
        super().__init__()
        self.reset()

    def reset(self, args=None):
        self.args = args if args is not None else {'B': -1, 'W': -1}

        self.board = -np.ones(6 * 6, dtype=np.int32)  # -1 is empty
        self.color = self.BLACK
        self.turn_count = 0  # -2 before setting original positions
        self.win_color = None
        self.piece_cnt = np.zeros(4, dtype=np.int32)
        self.board_index = -np.ones(6 * 6, dtype=np.int32)
        self.piece_position = np.zeros(2 * 8, dtype=np.int32)
        self.record = []

        b_pos, w_pos = self.args.get('B', -1), self.args.get('W', -1)
        self.set_pieces(self.BLACK, b_pos if b_pos >= 0 else random.randrange(70))
        self.set_pieces(self.WHITE, w_pos if w_pos >= 0 else random.randrange(70))

    def put_piece(self, piece, pos, piece_idx):
        self.board[pos] = piece
        self.piece_position[piece_idx] = pos
        self.board_index[pos] = piece_idx
        self.piece_cnt[piece] += 1

    def remove_piece(self, piece, pos):
        self.board[pos] = -1
        piece_idx = self.board_index[pos]
        self.board_index[pos] = -1
        self.piece_position[piece_idx] = -1
        self.piece_cnt[piece] -= 1

    def move_piece(self, piece, pos_from, pos_to):
        self.board[pos_from] = -1
        self.board[pos_to] = piece
        piece_idx = self.board_index[pos_from]
        self.board_index[pos_from] = -1
        self.board_index[pos_to] = piece_idx
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

    def xy2pos(self, x, y):
        return x * 6 + y

    def pos2x(self, pos):
        return np.sign(pos) * (abs(pos) // 6)

    def pos2y(self, pos):
        return np.sign(pos) * (abs(pos) % 6)

    def onboard(self, pos):
        return 0 <= pos < 6 * 6

    def onboard_xy(self, x, y):
        return 0 <= x < 6 and 0 <= y < 6

    def onboard_to(self, pos, d):
        return self.onboard_xy(
            self.pos2x(pos) + self.pos2x(self.D[d]),
            self.pos2y(pos) + self.pos2y(self.D[d])
        )

    def is_goal_action(self, c, pos_from, d):
        return self.position2str(pos_from) in self.GPOS[c] \
            and not self.onboard_to(pos_from, d) \
            and (d == 0 or d == 3)

    def colortype2piece(self, c, t):
        return c * 2 + t

    def piece2color(self, p):
        return -1 if p == -1 else p // 2

    def piece2type(self, p):
        return -1 if p == -1 else p % 2

    def str2piece(self, s):
        return self._P[s]

    def position2str(self, pos):
        if pos == -1:
            return '**'
        return self.X[self.pos2x(pos)] + self.Y[self.pos2y(pos)]

    def str2position(self, s):
        if s == '**':
            return -1
        return self.xy2pos(self.X.find(s[0]), self.Y.find(s[1]))

    def fromdirection2action(self, pos_from, d, c):
        if c == self.WHITE:
            pos_from = 35 - pos_from
            d = 3 - d
        return d * 36 + pos_from

    def action2from(self, a, c):
        pos = a % 36
        if c == self.WHITE:
            pos = 35 - pos
        return pos

    def action2direction(self, a, c):
        d = a // 36
        if c == self.WHITE:
            d = 3 - d
        return d

    def action2to(self, a, c):
        pos_from = self.action2from(a, c)
        d = self.action2direction(a, c)
        if self.onboard_to(pos_from, d):
            return pos_from + self.D[d]
        else:
            return -1

    def action2str(self, a, player):
        c = player
        pos_from = self.action2from(a, c)
        pos_to = self.action2to(a, c)
        return self.position2str(pos_from) + self.position2str(pos_to)

    def str2action(self, s, player):
        c = player
        pos_from = self.str2position(s[:2])
        pos_to = self.str2position(s[2:])

        if pos_to == -1:
            # it should arrive at a goal
            for d in [0, 3]:
                if not self.onboard_to(pos_from, d):
                    break
        else:
            # check action direction
            for d, dd in enumerate(self.D):
                if pos_to == pos_from + dd:
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
            s += self.X[i] + ' ' + ' '.join([self.P[self.board[self.xy2pos(i, j)]] for j in range(6)]) + '\n'
        s += 'color = ' + self.C[self.color] + '\n'
        s += 'record = ' + self.record_string()
        return s

    def play(self, action):
        # state transition
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr, self.turn()))
            return

        if self.turn_count < 0:
            self.set_pieces(self.color, action)
            self.color = self.opponent(self.color)
            self.turn_count += 1

        opos = self.action2from(action, self.color)
        npos = self.action2to(action, self.color)
        piece = self.board[opos]

        if npos == -1:
            # finish by goal
            self.remove_piece(piece, opos)
            self.win_color = self.color
        else:
            piece_cap = self.board[npos]
            if piece_cap != -1:
                # captupe opponent piece
                self.remove_piece(piece_cap, npos)
                if self.piece_cnt[piece_cap] == 0:
                    if self.piece2type(piece_cap) == self.BLUE:
                        # win by capturing all opponent blue pieces
                        self.win_color = self.color
                    else:
                        # lose by capturing all opponent red pieces
                        self.win_color = self.opponent(self.color)

            # move piece
            self.move_piece(piece, opos, npos)

        self.color = self.opponent(self.color)
        self.turn_count += 1
        self.record.append(action)

        if self.turn_count >= 200 and self.win_color is None:
            self.win_color = 2  # draw

    def diff_info(self):
        if len(self.record) == 0:
            return self.args
        return self.action2str(self.record[-1], (self.turn_count - 1) % 2)

    def reset_info(self, info):
        self.reset(info)

    def chance_info(self, _):
        pass

    def play_info(self, info):
        if info != "":
            self.play(info)

    def turn(self):
        return self.players()[self.turn_count % 2]

    def terminal(self):
        # check whether terminal state or not
        return self.win_color is not None

    def reward(self):
        # return terminal rewards
        rewards = [0, 0]
        if self.win_color == self.BLACK:
            rewards = [1, -1]
        elif self.win_color == self.WHITE:
            rewards = [-1, 1]
        return {p: rewards[idx] for idx, p in enumerate(self.players())}

    def legal(self, action):
        if self.turn_count < 0:
            return 0 <= action and action < 70
        pos_from = self.action2from(action, self.color)
        piece = self.board[pos_from]
        c, t = self.piece2color(piece), self.piece2type(piece)
        if c != self.color:
            # no piece on destination position
            return False

        return self._legal(c, t, pos_from, d)

    def _legal(self, c, t, pos_from, d):
        if self.onboard_to(pos_from, d):
            pos_to = pos_from + self.D[d]
            # can move to cell if there isn't my piece
            piece_cap = self.board[pos_to]
            return self.piece2color(piece_cap) != c
        else:
            # can move to my goal
            return t == self.BLUE and self.is_goal_action(c, pos_from, d)

    def legal_actions(self):
        # return legal action list
        if self.turn_count < 0:
            return [i for i in range(70)]
        actions = []
        for pos in self.piece_position[self.color*8:(self.color+1)*8]:
            if pos == -1:
                continue
            t = self.piece2type(self.board[pos])
            for d in range(4):
                if self._legal(self.color, t, pos, d):
                    action = self.fromdirection2action(pos, d, self.color)
                    actions.append(action)

        return actions

    def action_length(self):
        # maximul action label (it determines output size of policy function)
        return 70 if self.turn_count < 0 else 4 * 6 * 6

    def players(self):
        return [0, 1]

    def observation(self, player=None):
        # state representation to be fed into neural networks
        if self.turn_count < 0:
            return np.array([1 if self.color == self.BLACK else 0], dtype=np.float32)

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
            *[(1 if nropp   == i else 0) for i in range(1, 5)],
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
            red_o  if player is None else np.zeros_like(self.board),
        ]).reshape(-1, 6, 6).astype(np.float32)

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
        print(e.reward())
