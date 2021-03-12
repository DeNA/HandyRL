### Train AI Model for Tic-Tac-Toe

This section shows the training a model for [Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe). Tic-Tac-Toe is a very simple game. You can play by googling "Tic-Tac-Toe".

#### Step 1: Set up configuration

Set `config.yaml` for your training configuration. When you run a training with Tic-Tac-Toe and batch size 64, set like the following:


```yaml
env_args:
    env: 'TicTacToe'

train_args:
    ...
    batch_size: 64
    ...
```

NOTE: TicTacToe is used as a default game. [Here is the list of games](https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/). All parameters are shown in [Config Parameters](../documentation/parameters.md).


#### Step 2: Train!

After creating the configuration, you can start training by running the following command. The trained models are saved in `models` folder every `update_episodes` described in `config.yaml`.

```
python main.py --train
```


#### Step 3: Evaluate

After training, you can evaluate the model against any models. The below code evaluate the model of epoch 1 for 100 games with 4 processes.


```
python main.py --eval models/1.pth 100 4
```

NOTE: Default opponent AI is random agent implemented in `evaluation.py`. You can change the agent with any of your agents.