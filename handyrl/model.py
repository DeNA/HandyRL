# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# neural nets

import numpy as np
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F

from .util import map_r


def to_torch(x, transpose=False, unsqueeze=None):
    if x is None:
        return None
    elif isinstance(x, (list, tuple, set)):
        return type(x)(to_torch(xx, transpose, unsqueeze) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, to_torch(xx, transpose, unsqueeze)) for key, xx in x.items())

    a = np.array(x)
    if transpose:
        a = np.swapaxes(a, 0, 1)
    if unsqueeze is not None:
        a = np.expand_dims(a, unsqueeze)

    return torch.from_numpy(a).contiguous()


def to_numpy(x):
    return map_r(x, lambda x: x.detach().numpy() if x is not None else None)


def to_gpu(data):
    return map_r(data, lambda x: x.cuda() if x is not None else None)


# model base class

class BaseModel(nn.Module):
    def __init__(self, env, args=None):
        super().__init__()
        self.action_length = env.action_length()

    def init_hidden(self, batch_size=None):
        return None

    def inference(self, x, hidden, **kwargs):
        # numpy array -> numpy array
        self.eval()
        with torch.no_grad():
            xt = to_torch(x, unsqueeze=0)
            ht = to_torch(hidden, unsqueeze=0)
            outputs = self.forward(xt, ht, **kwargs)
        return map_r(outputs, lambda o: o.detach().numpy().squeeze(0) if o is not None else None)


# simple model

class RandomModel(BaseModel):
    def inference(self, x=None, hidden=None):
        return {'policy': np.zeros(self.action_length, dtype=np.float32), 'value': np.zeros(1, dtype=np.float32)}
