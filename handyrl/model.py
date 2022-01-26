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

    def init_hidden(self, batch_size=None):
        if hasattr(self.model, 'init_hidden'):
            if batch_size is None:  # for inference
                hidden = self.model.init_hidden([])
                return map_r(hidden, lambda h: h.detach().numpy() if isinstance(h, torch.Tensor) else h)
            else:  # for training
                return self.model.init_hidden(batch_size)
        return None

    def forward(self, x, hidden, **kwargs):
        if self.model.forward.__code__.co_argcount == 1 + 1:
            # ignore hidden state inputs if the number of arguments is just one
            assert len(kwargs) == 0
            return self.model.forward(x)
        else:
            # otherwise, users should prepare an argument for hidden states
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
    def __init__(self, model, x):
        super().__init__()
        wrapped_model = ModelWrapper(model)
        hidden = wrapped_model.init_hidden()
        outputs = wrapped_model.inference(x, hidden)
        self.output_dict = {key: np.zeros_like(value) for key, value in outputs.items()}

    def inference(self, *args):
        return self.output_dict
