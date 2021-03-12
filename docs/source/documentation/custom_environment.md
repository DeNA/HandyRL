## Customized Environment

In this page, you will know how to prepare the customized environment where you want to train an AI.

The basic steps are as below:
* Prepare the customized environment
* Prepare a config
* Train and evaluate on the customized environment

### Prepare the Customized Environment

To use the customized environment, write a wrapper class named `Environment` in `my_env.py` according to the HandyRL's format

**NOTE**: the games HandyRL supports are as below:
* turn-based game (alternating game) ([tictactoe.py](/handyrl/envs/tictactoe.py), [geister.py](/handyrl/envs/geister.py))
* simultaneous game ([hungry_geese.py](/handyrl/envs/kaggle/hungry_geese.py))

To see all interfaces of environment, check [environment.py](/handyrl/environment.py).

#### Turn-Based Game

Let's create a class `Environment`. We recommend to check compared with the implementation of sample environment like sample game [Tic-Tac-Toe](/handyrl/envs/tictactoe.py).

```python
# my_env.py
class Environment:
    def __init__(self, args={}):
        ...
```

Next, implement `reset()` and `play()` methods. `reset()` resets the game and `play()` steps the game state with one step.

```python
    #
    # Should be defined in all games
    #
    def reset(self, args={}):
        ...

    #
    # Should be defined in all games except you implement original step() function
    #
    def play(self, action, player):
        ...
```

`terminal()` returns whether the game finished or not
```py
    #
    # Should be defined in all games
    #
    def terminal(self):
        return False
```

`players()` returns the list of player id, and `turn()` returns the current player id.
```py
    #
    # Should be defined if you use multiplayer game or add name to each player
    #
    def players(self):
        return [0]

    #
    # Should be defined if you use multiplayer sequential action game
    #
    def turn(self):
        return 0
```

`reward()` and `outcome()` return the reward and outcome respectively at the step.
```py
    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        return {0: 0}

    #
    # Should be defined in all games
    #
    def outcome(self):
        return [0]
```

`action_length()` returns the length of all legal actions. `legal_actions()` returns the available actions in the step. Note that the actions are represented by int.
```py
    #
    # Should be defined in all games
    #
    def action_length(self):
        return 10

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        return 0
```

Finally, the features to feed neural network is implemented in `observation()`. This method need to return the input array for neural network.
```py
    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        return [0, 1, 2, 3]
```

#### Simultaneous Game

In simultaneous game, you need to implement different methods for handling multi players. We recommend to check compared with the implementation of sample environment like sample game [hungry_geese.py](/handyrl/envs/kaggle/hungry_geese.py).

`step()` method is required to handle the actions of multi players.
```py
    #
    # Should be defined in games which has simultaneous transition
    #
    def step(self, actions):
        ...
```

`turns()` returns the list of player id that can act in the step.
```py
    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [0, 1, 2, 3]
```


### Prepare a config

Set the name of the environment to `env` and script path to `source`.

```yaml
env_args:
    env: 'MyCustomizedEnvironment'
    source: 'my_env'
```

**NOTE**: the above config requires that `my_env.py` is in the same directory of `main.py`.

### Train and Evaluate on the Customized Environment

```
python main.py --train
```

```
python main.py --eval models/1.pth 100 4
```
