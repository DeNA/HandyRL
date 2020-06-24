# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# neural nets

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D
from tensorflow.keras import Model


def to_tensor(x, transpose=False):
    if x is None:
        return None
    elif isinstance(x, tf.Tensor):
        return x

    a = np.array(x)
    if transpose:
        a = np.swapaxes(a, 0, 1)

    if a.dtype == np.int32 or a.dtype == np.int64:
        a = a.astype(np.int64)
    else:
        a = a.astype(np.float32)

    return a


def to_numpy(x):
    if x is None:
        return None
    elif isinstance(x, tf.Tensor):
        a = x.numpy()
    elif isinstance(x, np.ndarray):
        a = x
    elif isinstance(x, tuple):
        return tuple(to_numpy(xx) for xx in x)
    else:
        a = np.array(x)
    return a


def to_gpu(data):
    return data


def to_gpu_or_not(data, gpu):
    return to_gpu(data) if gpu else data


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1))
    return x / x.sum(axis=-1)


def reload_model(x):
    weights = x
    from environments.tictactoe import Environment
    env = Environment()
    model = DuelingNet(env, {})
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
            h = h.view(h.size(0), -1)
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


'''class ConvLSTMCell(Model):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def init_hidden(self, input_size, batch_size):
        return tuple(
            torch.zeros(*batch_size, self.hidden_dim, *input_size),
            torch.zeros(*batch_size, self.hidden_dim, *input_size),
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=-3)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=-3)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class DRCCore(Model):
    def __init__(self, num_layers, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.num_layers = num_layers

        blocks = []
        for _ in range(self.num_layers):
            blocks.append(ConvLSTMCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                kernel_size=(kernel_size, kernel_size),
                bias=bias)
            )
        self.blocks = blocks

    def init_hidden(self, input_size, batch_size):
        hs, cs = [], []
        for block in self.blocks:
            h, c = block.init_hidden(input_size, batch_size)
            hs.append(h)
            cs.append(c)

        return torch.stack(hs), torch.stack(cs)

    def forward(self, x, hidden, num_repeats):
        if hidden is None:
            hidden = self.init_hidden(x.shape[-2:], x.shape[:-3])

        hs = [hidden[0][i] for i in range(self.num_layers)]
        cs = [hidden[1][i] for i in range(self.num_layers)]
        for _ in range(num_repeats):
            for i, block in enumerate(self.blocks):
                hs[i], cs[i] = block(x, (hs[i], cs[i]))

        return hs[-1], (torch.stack(hs), torch.stack(cs))'''


# simple model

class BaseModel(Model):
    def __init__(self, env, args=None, action_length=None):
        super().__init__()
        self.action_length = env.action_length() if action_length is None else action_length

    def init_hidden(self, batch_size=None):
        return None

    def inference(self, x, hidden=None, **kwargs):
        # numpy array -> numpy array
        xt = tuple(np.expand_dims(xx, 0) for xx in x)
        ht = tuple(np.expand_dims(hh, 1) for hh in hidden) if hidden is not None else None
        outputs = self.call(xt, ht, **kwargs)

        return tuple(
            [tf.squeeze(o, 0).numpy() for o in outputs[:-1]] + \
            [tuple(tf.squeeze(o, 1).numpy() for o in outputs[-1]) if outputs[-1] is not None else None]
        )


class RandomModel(BaseModel):
    def call(self, x=None, hidden=None):
        return tf.zeros((1, self.action_length)), tf.zeros((1, 1)), None


'''class LinearModel(BaseModel):
    def __init__(self, env, args=None, action_length=None):
        super().__init__(env, args, action_length)
        self.fc_p = Dense(self.action_length, use_bias=True)
        self.fc_v = Dense(use_bias=True)

    def call(self, x, hidden=None):
        return self.fc_p(x), self.fc_v(x), None'''


class DuelingNet(BaseModel):
    def __init__(self, env, args={}):
        super().__init__(env, args)

        self.input_size = env.observation()[0].shape

        layers, filters = args.get('layers', 3), args.get('filters', 32)
        internal_size = (filters, *self.input_size[1:])

        self.encoder = Encoder(self.input_size, filters)
        self.body = WideResNet(layers, filters)
        self.head_p = Head(internal_size, 2, self.action_length)
        self.head_v = Head(internal_size, 1, 1)

    def call(self, x, hidden=None):
        h = self.encoder(x[0])
        h = self.body(h)
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        return h_p, tf.nn.tanh(h_v), None


'''class DRC(BaseModel):
    def __init__(self, env, args={}, action_length=None):
        super().__init__(env, args, action_length)
        self.input_size = env.observation()[0].shape

        layers, filters = args.get('layers', 3), args.get('filters', 32)
        internal_size = (filters, *self.input_size[1:])

        self.encoder = Encoder(self.input_size, filters)
        self.body = DRCCore(layers, filters, filters)
        self.head_p = Head(internal_size, 2, self.action_length)
        self.head_v = Head(internal_size, 1, 1)

    def init_hidden(self, batch_size=None):
        if batch_size is None:  # for inference
            with torch.no_grad():
                return to_numpy(self.body.init_hidden(self.input_size[1:], []))
        else:  # for training
            return self.body.init_hidden(self.input_size[1:], batch_size)

    def call(self, x, hidden, num_repeats=1):
        h = self.encoder(x[0])
        h, hidden = self.body(h, hidden, num_repeats)
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        return h_p, tf.nn.tanh(h_v), hidden'''


class ModelCongress:
    def __init__(self, models):
        self.models = models

    def init_hidden(self, batch_size=None):
        return [m.init_hidden(batch_size) for m in self.models]

    def call(self, x, hiddens):
        # conmputes mean value of outputs
        ps, vs, nhiddens = [], [], []
        for i, model in enumerate(self.models):
            p, v, nhidden = model(x, hiddens[i])
            ps.append(softmax(p))
            vs.append(v)
            nhiddens.append(nhidden)
        return np.log(np.mean(ps, axis=0) + 1e-8), np.mean(vs, axis=0), nhiddens
