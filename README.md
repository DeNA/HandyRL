# HandyDistributedRL

![](https://github.com/DeNA/HandyDistributedRL/workflows/pytest/badge.svg?branch=master)

**HandyDistributedRL is a handy and simple framework for distributed reinforcement learning that is applicable to your own environments.**

Note: Python2 is not supported.

## Dependencies
```shell
pip install -r requirements.txt
```

## How to use

### configuration

Set `config.yaml` for your own configuration.

### Training

```shell
python train.py
```

### Evaluation

```shell
python evaluation.py
```

## How to use (for large scale training)

Set `'remote'` as `True` in `config.yaml`.
If you use remote machines as worker clients, you can set worker configuation in each client.

```shell
python train.py
```

In another window,
```shell
python worker.py
```

## Using your own environments

Write wrapper class named `Environment` following the format of the sample environment code `environments/tictactoe.py` or `environments/geister.py`.

```python
from environment import BaseEnvironment

class Environment(BaseEnvironment):
    def reset(self, args):
        ...
    def play(self, action):
        ...
    def terminal(self):
        ...
    def reward(self, player=-1):
        ...
    def legal_actions(self):
        ...
    def action_length(self):
        ...
    def observation(self, player=-1):
        ...
```

Set `'env'` and `'source'` variables in `config.yaml` as the name and path of your own environment.
