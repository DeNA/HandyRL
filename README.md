![HandyRL](logo.png)

![](https://github.com/DeNA/HandyRL/workflows/pytest/badge.svg?branch=master)

**HandyRL is a handy and simple framework for distributed reinforcement learning that is applicable to your own environments.**


# Quick Start, Easy to Win

*   Prepare your own environment
*   Let’s start large-scale distributed training
*   Get your great model!

## How to use


### Step 0: Install dependencies

HandyRL supports Python3.7+.
You need to install additional libraries (e.g. numpy, pytorch).
```
pip3 install -r requirements.txt
```

To use games of kaggle environments (e.g. Hungry Geese) you can install also additional dependencies.
```
pip3 install -r handyrl/envs/kaggle/requirements.txt
```

### Step 1: Set up configuration

Set `config.yaml` for your training configuration.
If you run a training with TicTacToe and batch size 64, set like the following:


```yaml
env_args:
    env: 'TicTacToe'
    source: 'handyrl.envs.tictactoe'

train_args:
    ...
    batch_size: 64
    ...
```

NOTE: TicTacToe is used as a default game. [Here is the list of games](handyrl/envs). When you use your own environment, set the name of the environment to `env` and script path to `source`.


### Step 2: Train!


```
python main.py --train
```

NOTE: Trained model is saved in `models` folder.


### Step 3: Evaluate

After training, you can evaluate the model against any models. The below code runs the evaluation for 100 games with 4 processes.


```
python main.py --eval models/1.pth 100 4
```


NOTE: Default opponent AI is random agent implemented in `evaluation.py`. You can change the agent with any of your agents.


## Distributed Training!

HandyRL allows you to learn a model remotely on a large scale.


### Step 1: Remote configuration

If you will use remote machines as worker clients(actors), you need to set training server(learner) address in each client:


```yaml
worker_args:
    server_address: '127.0.0.1'  # Set training server address to be connected from worker
    ...
```


NOTE: When you train a model on cloud(e.g. GCP, AWS), the internal/external IP of virtual machine can be set here.


### Step 2: Start training server


```
python main.py --train-server
```


NOTE: The server listens to connections from workers. The trained models are saved in `models` folder.


### Step 3: Start workers

After starting the training server, you can start the workers for data generation and evaluation.
In HandyRL, (multi-node) multiple workers can connect to the server.


```
python main.py --worker
```



### Step 4: Evaluate

After training, you can evaluate the model against any models. The below code runs the evaluation for 100 games with 4 processes.


```
python main.py --eval models/1.pth 100 4
```



## Custom environment

Write a wrapper class named `Environment` following the format of the sample environments.
The kind of your games are:
* turn-based game: see [tictactoe.py](handyrl/envs/tictactoe.py), [geister.py](handyrl/envs/geister.py)
* simultaneous game: see [hungry_geese.py](handyrl/envs/kaggle/hungry_geese.py)

To see all interfaces of environment, check [environment.py](handyrl/environment.py).



## Frequently Asked Questions


*   How to use rule-based AI as an opponent?
    *   You can easily use it by creating a rule-based AI method `rule_based_action()` in a class `Environment`.
*   How to change the opponent in evaluation?
    *   Set your agent in `evaluation.py` like `agents = [agent1, YourOpponentAgent()]`
* `too many open files` Error
    * This error happens in a large-scale training. You should increase the maximum file limit by running `ulimit -n 65536`. The value 65536 depends on a training setting. Note that the effect of `ulimit` is session-based so you will have to either change the limit permanently (OS and version dependent) or run this command in your shell starting script.
    * In Mac OSX, you may need to change the system limit with `launchctl` before running `ulimit -n` (e.g. [How to Change Open Files Limit on OS X and macOS](https://gist.github.com/tombigel/d503800a282fcadbee14b537735d202c))


## Use cases

*   [The 5th solution in Google Research Football with Manchester City F.C.](https://www.kaggle.com/c/google-football/discussion/203412) (Kaggle)
*   [Baseline RL AI in Hungry Geese](https://www.kaggle.com/yuricat/smart-geese-trained-by-reinforcement-learning) (Kaggle)
