import random
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from handyrl.environment import BaseEnvironment



class FootballNet(nn.Module):
    class FootballHead(nn.Module):
        def __init__(self, units0, units1):
            super().__init__()
            self.fc = nn.Linear(units0, units1)
            self.bn = nn.BatchNorm1d(units1)
            self.head_p = nn.Linear(units1, 19, bias=False)
            self.head_v = nn.Linear(units1, 1, bias=False)
            self.head_r = nn.Linear(units1, 1, bias=False)

        def forward(self, x):
            h = F.relu_(self.bn(self.fc(x)))
            p = self.head_p(h)
            v = self.head_v(h)
            r = self.head_r(h)
            return {'policy': p, 'value': v, 'return': r}

    class CNNModel(nn.Module):
        def __init__(self, final_filters):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(53, 128, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.pool1 = nn.AdaptiveAvgPool2d((1, 11))
            self.conv2 = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 160, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(160),
                nn.Conv2d(160, 96, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.Conv2d(96, final_filters, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(final_filters),
            )
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = x['cnn_feature']
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            return x

    class ActionHistoryEncoder(nn.Module):
        def __init__(self, hidden_size=64, num_layers=2):
            super().__init__()
            self.action_emd = nn.Embedding(19, 8)
            self.rnn = nn.GRU(8, hidden_size, num_layers, batch_first=True)

        def forward(self, x):
            h = self.action_emd(x['action_history'])
            h = h.squeeze(dim=2)
            self.rnn.flatten_parameters()
            h, _ = self.rnn(h)
            return h

    def __init__(self):
        super().__init__()

        self.cnn = self.CNNModel(64)  # to control
        self.rnn = self.ActionHistoryEncoder(64, 2)
        self.head = self.FootballHead(157, 64)

    def forward(self, x, hidden):
        cnn_h = self.cnn(x)
        rnn_h = self.rnn(x)

        h = torch.cat([
            cnn_h.view(cnn_h.size(0), -1),
            rnn_h[:, -1, :],
            x['ball'],
            x['match'],
            x['control']], -1)
        o = self.head(h)

        return o


# feature
def feature_from_states(states, info, number):
    # observation list to input tensor

    HISTORY_LENGTH = 8

    obs_history_ = [s['observation'][number] for s in reversed(states[-HISTORY_LENGTH:])]
    obs_history = obs_history_ + [obs_history_[-1]] * (HISTORY_LENGTH - len(obs_history_))
    obs = obs_history[0]

    action_history_ = [s['action'][number] for s in reversed(states[-HISTORY_LENGTH:])]
    action_history = action_history_ + [0] * (HISTORY_LENGTH - len(action_history_ ))

    """
    ・left players (x)
    ・left players (y)
    ・right players (x)
    ・right players (y)
    ・ball (x)
    ・ball (y)
    ・left goal (x)
    ・left goal (y)
    ・right goal (x)
    ・right goal (y)
    ・active (x)
    ・active (y)

    ・left players (x) - right players (x)
    ・left players (y) - right players (y)
    ・left players (x) - ball (x)
    ・left players (y) - ball (y)
    ・left players (x) - goal (x)
    ・left players (y) - goal (y)
    ・left players (x) - active (x)
    ・left players (y) - active (y)

    ・left players direction (x)
    ・left players direction (y)
    ・right players direction (x)
    ・right players direction (y)
    ・left players direction (x) - right players direction (x)
    ・left players direction (y) - right players direction (y)
    """

    # left players
    obs_left_team = np.array(obs['left_team'])
    left_player_x = np.repeat(obs_left_team[:, 0][..., None], 11, axis=1)
    left_player_y = np.repeat(obs_left_team[:, 1][..., None], 11, axis=1)

    # right players
    obs_right_team = np.array(obs['right_team'])
    right_player_x = np.repeat(obs_right_team[:, 0][..., None], 11, axis=1).transpose(1, 0)
    right_player_y = np.repeat(obs_right_team[:, 1][..., None], 11, axis=1).transpose(1, 0)

    # ball
    obs_ball = np.array(obs['ball'])
    ball_x = np.ones((11, 11)) * obs_ball[0]
    ball_y = np.ones((11, 11)) * obs_ball[1]
    ball_z = np.ones((11, 11)) * obs_ball[2]

    # goal
    left_goal, right_goal = [-1, 0], [1, 0]
    left_goal_x = np.ones((11, 11)) * left_goal[0]
    left_goal_y = np.ones((11, 11)) * left_goal[1]
    right_goal_x = np.ones((11, 11)) * right_goal[0]
    right_goal_y = np.ones((11, 11)) * right_goal[1]

    # side line
    side_line_y = [-.42, .42]
    side_line_y_top = np.ones((11, 11)) * side_line_y[0]
    side_line_y_bottom = np.ones((11, 11)) * side_line_y[1]

    # active
    active = np.array(obs['active'])
    active_player_x = np.repeat(obs_left_team[active][0][..., None, None], 11, axis=1).repeat(11, axis=0)
    active_player_y = np.repeat(obs_left_team[active][1][..., None, None], 11, axis=1).repeat(11, axis=0)

    # left players - right players
    left_minus_right_player_x = obs_left_team[:, 0][..., None] - obs_right_team[:, 0]
    left_minus_right_player_y = obs_left_team[:, 1][..., None] - obs_right_team[:, 1]

    # left players - ball
    left_minus_ball_x = (obs_left_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1)
    left_minus_ball_y = (obs_left_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1)

    # left players - right goal
    left_minus_right_goal_x = (obs_left_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1)
    left_minus_right_goal_y = (obs_left_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1)

    # left players - left goal
    left_minus_left_goal_x = (obs_left_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1)
    left_minus_left_goal_y = (obs_left_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1)

    # right players - right goal
    right_minus_right_goal_x = (obs_right_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_right_goal_y = (obs_right_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1).transpose(1, 0)

    # right players - left goal
    right_minus_left_goal_x = (obs_right_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_left_goal_y = (obs_right_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1).transpose(1, 0)

    # left players (x) - active
    left_minus_active_x = (obs_left_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11, axis=1)
    left_minus_active_y = (obs_left_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11, axis=1)

    # right player - ball
    right_minus_ball_x = (obs_right_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_ball_y = (obs_right_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1).transpose(1, 0)

    # right player - active
    right_minus_active_x = (obs_right_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_active_y = (obs_right_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11, axis=1).transpose(1, 0)

    # left player - side line
    left_minus_side_top = np.abs(obs_left_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1)
    left_minus_side_bottom = np.abs(obs_left_team[:, 1][..., None] - side_line_y[1]).repeat(11, axis=1)

    # right player - side line
    right_minus_side_top = np.abs(obs_right_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_side_bottom = np.abs(obs_right_team[:, 1][..., None] - side_line_y[1]).repeat(11, axis=1).transpose(1, 0)

    # left players direction
    obs_left_team_direction = np.array(obs['left_team_direction'])
    left_player_direction_x = np.repeat(obs_left_team_direction[:, 0][..., None], 11, axis=1)
    left_player_direction_y = np.repeat(obs_left_team_direction[:, 1][..., None], 11, axis=1)

    # right players direction
    obs_right_team_direction = np.array(obs['right_team_direction'])
    right_player_direction_x = np.repeat(obs_right_team_direction[:, 0][..., None], 11, axis=1).transpose(1, 0)
    right_player_direction_y = np.repeat(obs_right_team_direction[:, 1][..., None], 11, axis=1).transpose(1, 0)

    # ball direction
    obs_ball_direction = np.array(obs['ball_direction'])
    ball_direction_x = np.ones((11, 11)) * obs_ball_direction[0]
    ball_direction_y = np.ones((11, 11)) * obs_ball_direction[1]
    ball_direction_z = np.ones((11, 11)) * obs_ball_direction[2]

    # left players direction - right players direction
    left_minus_right_player_direction_x = obs_left_team_direction[:, 0][..., None] - obs_right_team_direction[:, 0]
    left_minus_right_player_direction_y = obs_left_team_direction[:, 1][..., None] - obs_right_team_direction[:, 1]

    # left players direction - ball direction
    left_minus_ball_direction_x = (obs_left_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(11, axis=1)
    left_minus_ball_direction_y = (obs_left_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(11, axis=1)

    # right players direction - ball direction
    right_minus_ball_direction_x = (obs_right_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_ball_direction_y = (obs_right_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(11, axis=1).transpose(1, 0)

    # ball rotation
    obs_ball_rotation = np.array(obs['ball_rotation'])
    ball_rotation_x = np.ones((11, 11)) * obs_ball_rotation[0]
    ball_rotation_y = np.ones((11, 11)) * obs_ball_rotation[1]
    ball_rotation_z = np.ones((11, 11)) * obs_ball_rotation[2]

    cnn_feature = np.stack([
        left_player_x,
        left_player_y,
        right_player_x,
        right_player_y,
        ball_x,
        ball_y,
        ball_z,
        left_goal_x,
        left_goal_y,
        right_goal_x,
        right_goal_y,
        side_line_y_top,
        side_line_y_bottom,
        active_player_x,
        active_player_y,
        left_minus_right_player_x,
        left_minus_right_player_y,
        left_minus_right_goal_x,
        left_minus_right_goal_y,
        left_minus_left_goal_x,
        left_minus_left_goal_y,
        right_minus_right_goal_x,
        right_minus_right_goal_y,
        right_minus_left_goal_x,
        right_minus_left_goal_y,
        left_minus_side_top,
        left_minus_side_bottom,
        right_minus_side_top,
        right_minus_side_bottom,
        right_minus_ball_x,
        right_minus_ball_y,
        right_minus_active_x,
        right_minus_active_y,
        left_minus_ball_x,
        left_minus_ball_y,
        left_minus_active_x,
        left_minus_active_y,
        ball_direction_x,
        ball_direction_y,
        ball_direction_z,
        left_minus_ball_direction_x,
        left_minus_ball_direction_y,
        right_minus_ball_direction_x,
        right_minus_ball_direction_y,
        left_player_direction_x,
        left_player_direction_y,
        right_player_direction_x,
        right_player_direction_y,
        left_minus_right_player_direction_x,
        left_minus_right_player_direction_y,
        ball_rotation_x,
        ball_rotation_y,
        ball_rotation_z,
    ], axis=0).astype(np.float32)

    # ball
    BALL_OWEND_1HOT = {-1: [0, 0], 0: [1, 0], 1: [0, 1]}
    ball_owned_team_ = obs['ball_owned_team']
    ball_owned_team = BALL_OWEND_1HOT[ball_owned_team_]  # {-1, 0, 1} None, self, opponent
    PLAYER_1HOT = np.concatenate([np.eye(11), np.zeros((1, 11))])
    ball_owned_player_ = PLAYER_1HOT[obs['ball_owned_player']]  # {-1, N-1}
    if ball_owned_team_ == -1:
        my_ball_owned_player = PLAYER_1HOT[-1]
        op_ball_owned_player = PLAYER_1HOT[-1]
    elif ball_owned_team_ == 0:
        my_ball_owned_player = ball_owned_player_
        op_ball_owned_player = PLAYER_1HOT[-1]
    else:
        my_ball_owned_player = PLAYER_1HOT[-1]
        op_ball_owned_player = ball_owned_player_

    ball_features = np.concatenate([
        obs['ball'],
        obs['ball_direction'],
        obs['ball_rotation']
    ]).astype(np.float32)

    # self team
    left_team_features = np.concatenate([
        [[1] for _ in obs['left_team']],  # left team flag
        obs['left_team'],  # position
        obs['left_team_direction'],
        [[v] for v in obs['left_team_tired_factor']],
        [[v] for v in obs['left_team_yellow_card']],
        [[v] for v in obs['left_team_active']],
        my_ball_owned_player[...,np.newaxis]
    ], axis=1).astype(np.float32)

    left_team_indice = np.arange(0, 11, dtype=np.int32)

    # opponent team
    right_team_features = np.concatenate([
        [[0] for _ in obs['right_team']],  # right team flag
        obs['right_team'],  # position
        obs['right_team_direction'],
        [[v] for v in obs['right_team_tired_factor']],
        [[v] for v in obs['right_team_yellow_card']],
        [[v] for v in obs['right_team_active']],
        op_ball_owned_player[...,np.newaxis]
    ], axis=1).astype(np.float32)

    right_team_indice = np.arange(0, 11, dtype=np.int32)

    # distance information
    def get_distance(xy1, xy2):
        return (((xy1 - xy2) ** 2).sum(axis=-1)) ** 0.5

    def get_line_distance(x1, x2):
        return np.abs(x1 - x2)

    def multi_scale(x, scale):
        return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

    both_team = np.array(obs['left_team'] + obs['right_team'], dtype=np.float32)
    ball = np.array([obs['ball'][:2]], dtype=np.float32)
    goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
    goal_line_x = np.array([-1, 1], dtype=np.float32)
    side_line_y = np.array([-.42, .42], dtype=np.float32)

    # ball <-> goal, goal line, side line distance
    b2g_distance = get_distance(ball, goal)
    b2gl_distance = get_line_distance(ball[0][0], goal_line_x)
    b2sl_distance = get_line_distance(ball[0][1], side_line_y)
    b2o_distance = np.concatenate([
        b2g_distance, b2gl_distance, b2sl_distance
    ], axis=-1)

    # player <-> ball, goal, back line, side line distance
    p2b_distance = get_distance(both_team[:,np.newaxis,:], ball[np.newaxis,:,:])
    p2g_distance = get_distance(both_team[:,np.newaxis,:], goal[np.newaxis,:,:])
    p2gl_distance = get_line_distance(both_team[:,:1], goal_line_x[np.newaxis,:])
    p2sl_distance = get_line_distance(both_team[:,1:], side_line_y[np.newaxis,:])
    p2bo_distance = np.concatenate([
        p2b_distance, p2g_distance, p2gl_distance, p2sl_distance
    ], axis=-1)

    # player <-> player distance
    p2p_distance = get_distance(both_team[:,np.newaxis,:], both_team[np.newaxis,:,:])

    # controlled player information
    control_flag_ = np.array(PLAYER_1HOT[obs['active']], dtype=np.float32)
    control_flag = np.concatenate([control_flag_, np.zeros(len(obs['right_team']), dtype=np.float32)])[...,np.newaxis]

    # controlled status information
    DIR = [
        [-1, 0], [-.707, -.707], [0,  1], [ .707, -.707],  # L, TL, T, TR
        [ 1, 0], [ .707,  .707], [0, -1], [-.707,  .707]   # R, BR, B, BL
    ]

    sticky_direction = DIR[np.where(obs['sticky_actions'][:8] == 1)[0][0]] if 1 in obs['sticky_actions'][:8] else [0, 0]
    sticky_flags = obs['sticky_actions'][8:]

    control_features = np.concatenate([
        sticky_direction,
        sticky_flags,
    ]).astype(np.float32)

    # Match state
    if obs['steps_left'] > info['half_step']:
        steps_left_half = obs['steps_left'] - info['half_step']
    else:
        steps_left_half = obs['steps_left']
    match_features = np.concatenate([
        multi_scale(obs['score'], [1, 3]).ravel(),
        multi_scale(obs['score'][0] - obs['score'][1], [1, 3]),
        multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),
        multi_scale(steps_left_half, [10, 100, 1000, 10000]),
        ball_owned_team,
    ]).astype(np.float32)

    mode_index = np.array([obs['game_mode']], dtype=np.int32)

    action_history = np.array(action_history, dtype=np.int32)[..., None]

    return {
        # features
        'ball': ball_features,
        'match': match_features,
        'player': {
            'self': left_team_features,
            'opp': right_team_features
        },
        'control': control_features,
        'player_index': {
            'self': left_team_indice,
            'opp': right_team_indice
        },
        'mode_index': mode_index,
        'control_flag': control_flag,
        # distances
        'distance': {
            'p2p': p2p_distance,
            'p2bo': p2bo_distance,
            'b2o': b2o_distance
        },
        # CNN
        'cnn_feature': cnn_feature,
        'action_history': action_history
    }


# https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/football/helpers.py

import enum

class Action(enum.IntEnum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass= 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18

sticky_index_to_action = [
    Action.Left,
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.Right,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Sprint,
    Action.Dribble
]

action_to_sticky_index = {
    a: index for index, a in enumerate(sticky_index_to_action)
}

class PlayerRole(enum.IntEnum):
    GoalKeeper = 0
    CenterBack = 1
    LeftBack = 2
    RightBack = 3
    DefenceMidfield = 4
    CentralMidfield = 5
    LeftMidfield = 6
    RIghtMidfield = 7
    AttackMidfield = 8
    CentralFront = 9


class GameMode(enum.IntEnum):
    Normal = 0
    KickOff = 1
    GoalKick = 2
    FreeKick = 3
    Corner = 4
    ThrowIn = 5
    Penalty = 6


class Environment(BaseEnvironment):
    ACTION_LEN = 19
    CONTROLLED_PLAYERS = 1
    FINISH_BY_GOAL = True

    def __init__(self, args=None):
        self.env = None
        args = args if args is not None else {}
        self.limit_step = args.get('limit_step', 600)
        self.controlled_players = 1

    def reset(self, args=None):
        if self.env is None:
            from gfootball.env import create_environment

            self.env = create_environment(
                env_name="11_vs_11_stochastic",
                representation='raw',
                number_of_left_players_agent_controls=self.CONTROLLED_PLAYERS,
                number_of_right_players_agent_controls=self.CONTROLLED_PLAYERS,
                other_config_options={'action_set': 'v2'})

        obs = self.env.reset()
        self.update({'observation': obs, 'action': [0] * self.CONTROLLED_PLAYERS * 2}, reset=True)

    def update(self, state, reset):
        if reset:
            self.done = False
            self.prev_score = [0, 0]
            self.states = []
            self.half_step = 1500
            self.reserved_action = [None, None]
        else:
            self.prev_score = self.score()

        state = copy.deepcopy(state)
        state = self._preprocess_state(state)
        self.states.append(state)

        if reset:
            self.half_step = state['observation'][0]['steps_left'] // 2

    def step(self, actions):
        # state transition function
        # action is integer (0 ~ 18)
        actions = copy.deepcopy(actions)
        for i, res_action in enumerate(self.reserved_action):
            if res_action is not None:
                actions[i] = res_action

        # step environment
        flat_actions = [actions[0], actions[1]]
        obs, _, self.done, _ = self.env.step(flat_actions)
        self.update({'observation': obs, 'action': flat_actions}, reset=False)

    def diff_info(self):
        return self.states[-1]

    def turns(self):
        return self.players()

    def players(self):
        return [0, 1]

    def terminal(self):
        # check whether the state is terminal
        return self.done \
            or len(self.states) > self.limit_step \
            or (self.FINISH_BY_GOAL and sum(self.score().values()) > 0)

    def score(self):
        if len(self.states) == 0:
            return [0, 0]
        state = self.states[-1]
        return {p: state['observation'][0]['score'][p] for p in self.players()}

    def reward(self):
        prev_score = self.prev_score
        score = self.score()

        rewards = {}
        for p in self.players():
            r = 1.0 * (score[p] - prev_score[p]) - 1.0 * (score[1 - p] - prev_score[1 - p])
            rewards[p] = r

        return rewards

    def outcome(self):
        scores = self.score()
        if scores[0] > scores[1]:
            return {0: 1, 1: -1}
        elif scores[0] < scores[1]:
            return {0: -1, 1: 1}
        return {0: 0, 1: 0}

    def legal_actions(self, player, number=0):
        # legal action list
        return list(range(self.ACTION_LEN))

    def raw_observation(self, player):
        return self.states[-1]['observation'][player]

    def observation(self, player, number=0):
        # input feature for neural nets
        info = {'half_step': self.half_step}
        return feature_from_states(self.states, info, player * self.CONTROLLED_PLAYERS + number)

    def _preprocess_state(self, state):
        if state is None:
            return state

        # in ball-dead state, set ball owned player and team
        for o in state['observation']:
            mode = o['game_mode']
            if mode == GameMode.FreeKick or \
                mode == GameMode.Corner or \
                mode == GameMode.Penalty or \
                mode == GameMode.GoalKick:
                # find nearest player and team
                def dist(xy1, xy2):
                    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5
                team_player_position = [(0, i, p) for i, p in enumerate(o['left_team'])] + \
                    [(1, i, p) for i, p in enumerate(o['right_team'])]
                distances = [(t[0], t[1], dist(t[2], o['ball'][:2])) for t in team_player_position]
                distances = sorted(distances, key=lambda x: x[2])

                o['ball_owned_team'] = distances[0][0]
                o['ball_owned_player'] = distances[0][1]

        return state

    def rule_based_action(self, player=None, number=0, key=None):
        if key is None:
            key = 'builtin_ai'

        if key == 'builtin_ai':
            return 19
        elif key == 'idle':
            return 14
        elif key == 'right':
            return 5

    def net(self):
        return FootballNet()


if __name__ == '__main__':
    e = Environment()
    for _ in range(1):
        e.reset()
        o = e.observation(0)
        while not e.terminal():
            # print(e)
            _ = e.observation(0)
            _ = e.observation(1)
            #print(e.raw_observation(0)[0]['steps_left'])
            action_list = [0, 0]
            action_list[0] = random.choice(e.legal_actions(0))
            action_list[1] = 19
            print(len(e.states), action_list)
            e.step(action_list)
            print(e.score())
        print(e.outcome())
