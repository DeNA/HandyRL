# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Paper that proposed VTrace algorithm
# https://arxiv.org/abs/1802.01561
# Official code
# https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py

# algorithms and losses

from collections import deque

import torch


def monte_carlo(values, returns):
    return returns, returns - values


def temporal_difference(values, returns, rewards, lmb, gamma):
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        reward = rewards[:, i] if rewards is not None else 0
        target_values.appendleft(reward + gamma * ((1 - lmb) * values[:, i + 1] + lmb * target_values[0]))

    target_values = torch.stack(tuple(target_values), dim=1)

    return target_values, target_values - values


def upgo(values, returns, rewards, lmb, gamma):
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        value = values[:, i + 1]
        reward = rewards[:, i] if rewards is not None else 0
        target_values.appendleft(reward + gamma * torch.max(value, (1 - lmb) * value + lmb * target_values[0]))

    target_values = torch.stack(tuple(target_values), dim=1)

    return target_values, target_values - values


def vtrace(values, returns, rewards, lmb, gamma, rhos, cs):
    rewards = rewards if rewards is not None else 0
    values_t_plus_1 = torch.cat([values[:, 1:], returns[:, -1:]], dim=1)
    deltas = rhos * (rewards + gamma * values_t_plus_1 - values)

    # compute Vtrace value target recursively
    vs_minus_v_xs = deque([deltas[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        vs_minus_v_xs.appendleft(deltas[:, i] + gamma * lmb * cs[:, i] * vs_minus_v_xs[0])

    vs_minus_v_xs = torch.stack(tuple(vs_minus_v_xs), dim=1)
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat([vs[:, 1:], returns[:, -1:]], dim=1)
    advantages = rewards + gamma * vs_t_plus_1 - values

    return vs, advantages


def compute_target(algorithm, values, returns, rewards, lmb, gamma, rhos, cs):
    if values is None:
        # In the absence of a baseline, Monte Carlo returns are used.
        return returns, returns

    if algorithm == 'MC':
        return monte_carlo(values, returns)
    elif algorithm == 'TD':
        return temporal_difference(values, returns, rewards, lmb, gamma)
    elif algorithm == 'UPGO':
        return upgo(values, returns, rewards, lmb, gamma)
    elif algorithm == 'VTRACE':
        return vtrace(values, returns, rewards, lmb, gamma, rhos, cs)
    else:
        print('No algorithm named %s' % algorithm)
