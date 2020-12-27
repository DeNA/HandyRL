# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# neural nets

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D


from .util import map_r


def to_tensor(x, transpose=False, unsqueeze=None):
    if x is None:
        return None
    elif isinstance(x, (list, tuple, set)):
        return type(x)(to_tensor(xx, transpose, unsqueeze) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, to_tensor(xx, transpose, unsqueeze)) for key, xx in x.items())

    a = np.array(x)
    if transpose:
        a = np.swapaxes(a, 0, 1)
    if unsqueeze is not None:
        a = np.expand_dims(a, unsqueeze)

    if a.dtype == np.int32 or a.dtype == np.int64:
        a = a.astype(np.int64)
    else:
        a = a.astype(np.float32)

    return a


def to_numpy(x):
    return map_r(x, lambda x: x.numpy() if x is not None else None)


def to_gpu(data):
    return data


def to_gpu_or_not(data, gpu):
    return to_gpu(data) if gpu else data


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1))
    return x / x.sum(axis=-1)


def reload_model(x, env, args):
    model_class, weights = x
    model = model_class(env, args)
    model.inference(env.observation())
    model.set_weights(weights)
    return model


class ConvBN(Layer):
    def __init__(self, filters, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = Conv2D(filters, kernel_size, padding='same', use_bias=bias)
        self.bn = tf.keras.layers.BatchNormalization() if bn else None

    def call(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class DenseBN(Layer):
    def __init__(self, units, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.dense = Dense(units, use_bias=bias)
        self.bn = tf.keras.layers.BatchNormalization() if bn else None

    def call(self, x):
        h = self.dense(x)
        if self.bn is not None:
            size = h.size()
            h = h.view(-1, self.bnunits)
            h = self.bn(h)
            h = h.view(*size)
        return h


class WideResidualBlock(Layer):
    def __init__(self, filters, kernel_size, bn):
        super().__init__()
        self.conv1 = ConvBN(filters, kernel_size, bn)
        self.conv2 = ConvBN(filters, kernel_size, bn)

    def call(self, x):
        return tf.nn.relu(x + self.conv2(tf.nn.relu(self.conv1(x))))


class WideResNet(Layer):
    def __init__(self, blocks, filters):
        super().__init__()
        self.blocks = [WideResidualBlock(filters, 3, bn=False) for _ in range(blocks)]

    def call(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return h


class Encoder(Layer):
    def __init__(self, input_size, filters):
        super().__init__()
        self.conv = Conv2D(filters, 3, padding='same')

    def call(self, x):
        return tf.nn.leaky_relu(self.conv(x), 0.1)


class Head(Layer):
    def __init__(self, input_size, filters, outputs):
        super().__init__()

        self.board_size = input_size[1] * input_size[2]
        self.filters = filters

        self.conv = Conv2D(filters, 1, padding='same')
        self.fc = Dense(outputs, use_bias=False)

    def call(self, x):
        h = tf.nn.leaky_relu(self.conv(x), 0.1)
        h = tf.reshape(h, [-1, self.board_size * self.filters])
        h = self.fc(h)
        return h


class DRC(tf.keras.Model):
    def __init__(self, num_layers, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.num_layers = num_layers

        self.blocks = []
        for _ in range(self.num_layers):
            blocks.append(tf.keras.ConvLSTM2D(
                filters=hidden_dim,
                kernel_size=(kernel_size, kernel_size),
                use_bias=bias
            ))

    def call(self, x, hidden, num_repeats):
        if hidden is None:
            hidden = self.reset_states(x.shape[-2:], x.shape[:-3])

        hs, cs = hidden
        for _ in range(num_repeats):
            for i, block in enumerate(self.blocks):
                hs[i], cs[i] = block(x, (hs[i], cs[i]))

        return hs[-1], (hs, cs)


class BaseModel(tf.keras.Model):
    def __init__(self, env, args=None, action_length=None):
        super().__init__()
        self.action_length = env.action_length() if action_length is None else action_length

    def init_hidden(self, batch_size=None):
        return None

    def inference(self, x, hidden=None, **kwargs):
        # numpy array -> numpy array
        xt = to_tensor(x, unsqueeze=0)
        ht = to_tensor(hidden, unsqueeze=0)
        outputs = self.call(xt, ht, **kwargs)

        return tuple(
            [(to_numpy(o).squeeze(0) if o is not None else None) for o in outputs[:-1]] +
            [map_r(outputs[-1], lambda o: to_numpy(o).squeeze(0)) if outputs[-1] is not None else None]
        )


# simple models

class RandomModel(BaseModel):
    def call(self, x=None, hidden=None):
        return tf.zeros((1, self.action_length)), tf.zeros((1, 1)), None, None


class DuelingNet(BaseModel):
    def __init__(self, env, args={}):
        super().__init__(env, args)

        self.input_size = env.observation().shape

        layers, filters = args.get('layers', 3), args.get('filters', 32)
        internal_size = (filters, *self.input_size[1:])

        self.encoder = Encoder(self.input_size, filters)
        self.body = WideResNet(layers, filters)
        self.head_p = Head(internal_size, 2, self.action_length)
        self.head_v = Head(internal_size, 1, 1)

    def call(self, x, hidden=None):
        h = self.encoder(x)
        h = self.body(h)
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        return h_p, tf.tanh(h_v), None, None
