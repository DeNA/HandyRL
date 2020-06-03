# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# game environment

import os
import sys
import copy
import importlib

import numpy as np

sys.path.append(os.path.dirname(__file__))

_envname = None
_env_module = None


def prepare(env_args):
    envname = env_args['env']
    env_source = env_args['source']

    global _envname
    global _env_module

    _envname = envname
    _env_module = importlib.import_module(env_source)

    if _env_module is None:
        print("No environment %s" % envname)
    elif hasattr(_env_module, 'prepare'):
        _env_module.prepare()


def make(args=None):
    global _envname
    global _env_module

    if _env_module is None:
        print("No environment %s" % _envname)
    else:
        if hasattr(_env_module, 'default_env_args'):
            env_args = copy.deepcopy(_env_module.default_env_args)
        else:
            env_args = {}

        if args is not None and 'id' in args:
            env_args['id'] = args['id']
        return _env_module.Environment(env_args)


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
    def reward(self, player=-1):
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
    # Should be defined in all games
    #
    def observation(self, player=-1):
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
