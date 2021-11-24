## Train with Customized Environment

In this page, you will know how to prepare the customized environment where you want to train an AI.

The basic steps are as below:
* Prepare the customized environment
* Prepare your config file
* Train and evaluate on the customized environment

### Prepare the Customized Environment

To use the customized environment, write a wrapper class named `Environment` in `my_env.py` according to the HandyRL's format.

HandyRL currently supports games like below:
* Turn-based game (alternating game) ([tictactoe.py](/handyrl/envs/tictactoe.py), [geister.py](/handyrl/envs/geister.py))
* Simultaneous game ([hungry_geese.py](/handyrl/envs/kaggle/hungry_geese.py))

To see all interfaces of environment, check [environment.py](/handyrl/environment.py).

#### Turn-Based Game

Let's create a class `Environment`. If it is the first time for you to use HandyRL, we recommend to check your script by comparing with the implementation of sample environment like sample game [Tic-Tac-Toe](/handyrl/envs/tictactoe.py).

```python
    # my_env.py
    class Environment:
        def __init__(self, args={}):
            ...
```

Next, implement `reset()` which resets the game. Then `play()` or `step()` methods should be implemented. `play()` and `step()` function step the game state with one step. Note that the difference between those methods is the argument.

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

    #
    # Should be defined in games which has simultaneous transition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)
```

`terminal()` returns whether the game finished or not.
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
        return [0, 1]  # player id

    #
    # Should be defined if you use multiplayer sequential action game
    #
    def turn(self):
        return 0  # player id
```

`reward()` and `outcome()` return the reward and outcome respectively at the step.
```py
    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        return {0: 1, 1: 10}  # {player_id: reward}

    #
    # Should be defined in all games
    #
    def outcome(self):
        return {0: -1, 1: 1}  # {player_id: outcome} -1: loss, 0: draw, 1: win

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        return [0, 3]  # right(0) and top(3) available
```

Finally, the features and neural network are implemented in `observation()` and `net()`. `observation()` returns the input array for neural network. `net()` returns PyTorch model.
```py
    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        obs = ...
        return np.array(obs)  # this array will be fed to neural network

    def net(self):
        return YourNetworkModuleClass
```

#### Simultaneous Game

As the difference between the turn-based game, you need to implement step() and turns(). we recommend to check your script by comparing with the implementation of sample environment like sample game [hungry_geese.py](/handyrl/envs/kaggle/hungry_geese.py).

First, `step()` method is required to handle the actions of multi players.
```py
    #
    # Should be defined in games which has simultaneous transition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)
```

Next, `turns()` returns the list of player id that can act in the turn.
```py
    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [0, 1, 2, 3]
```


### Prepare your config file

Set the name of the environment or path to environment script to `env`.

```yaml
env_args:
    env: 'path.to.my_env'
```

### Train and evaluate on the customized environment

```
python main.py --train
```

```
python main.py --eval models/1.pth 100 4
```
