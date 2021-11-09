# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# neural nets

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F

from .util import map_r


def to_torch(x):
    return map_r(x, lambda x: torch.from_numpy(np.array(x)).contiguous() if x is not None else None)


def to_numpy(x):
    return map_r(x, lambda x: x.detach().numpy() if x is not None else None)


def to_gpu(data):
    return map_r(data, lambda x: x.cuda() if x is not None else None)


# model wrapper class

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        def get_argument_names(f):
            return f.__code__.co_varnames[:f.__code__.co_argcount]
        self.forward_args = get_argument_names(self.model.forward)

    def init_hidden(self, batch_size=None):
        if hasattr(self.model, 'init_hidden'):
            return self.model.init_hidden(batch_size)
        return None

    def forward(self, x, hidden, **kwargs):
        # Remove 'hidden' input if it will not accepted
        if 'hidden' not in self.forward_args:
            return self.model.forward(x, **kwargs)
        else:
            return self.model.forward(x, hidden, **kwargs)

    def inference(self, x, hidden, **kwargs):
        # numpy array -> numpy array
        if hasattr(self.model, 'inference'):
            return self.model.inference(x, hidden, **kwargs)

        self.eval()
        with torch.no_grad():
            xt = map_r(x, lambda x: torch.from_numpy(np.array(x)).contiguous().unsqueeze(0) if x is not None else None)
            ht = map_r(hidden, lambda h: torch.from_numpy(np.array(h)).contiguous().unsqueeze(0) if h is not None else None)
            outputs = self.forward(xt, ht, **kwargs)
        return map_r(outputs, lambda o: o.detach().numpy().squeeze(0) if o is not None else None)


# simple model

class RandomModel(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.action_length = env.action_length()

    def inference(self, x=None, hidden=None):
        return {
            'policy': np.zeros(self.action_length, dtype=np.float32),
            'value': np.zeros(1, dtype=np.float32),
        }
