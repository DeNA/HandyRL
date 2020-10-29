# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# game environment

import os
import sys
import importlib


sys.path.append(os.path.dirname(__file__))


def prepare(env_args):
    env_name = env_args['env']
    env_source = env_args['source']

    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    elif hasattr(env_module, 'prepare'):
        env_module.prepare()


def make(env_args):
    env_name = env_args['env']
    env_source = env_args['source']

    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    else:
        return env_module.Environment(env_args)


# base class of Environment

class BaseEnvironment:
    def __init__(self, args={}):
        self.reset()

    def __str__(self):
        return ''

    #
    # Should be defined in all games
    #
    def reset(self, args={}):
        raise NotImplementedError()

    #
    # Should be defined in all games which has stochastic state transition before deciding action
    #
    def chance(self):
        pass

    #
    # Should be defined in all games
    #
    def play(self, action):
        raise NotImplementedError()

    #
    # Should be defined if you use multiplayer game
    #
    def turn(self):
        return 0

    #
    # Should be defined in all games
    #
    def terminal(self):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def reward(self):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def legal_actions(self):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def action_length(self):
        raise NotImplementedError()

    #
    # Should be defined if you use multiplayer game or add name to each player
    #
    def players(self):
        return [0]

    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        raise NotImplementedError()

    #
    # Should be defined if you use network battle mode
    #
    def diff_info(self):
        return ''

    #
    # Should be defined if you use network battle mode
    #
    def reset_info(self, _):
        self.reset()

    #
    # Should be defined if you use network battle mode
    #
    def chance_info(self, _):
        self.chance()

    #
    # Should be defined if you use network battle mode
    #
    def play_info(self, info):
        self.play(info)
