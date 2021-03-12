===========================
Customized Environment
===========================

In this page, you will know how to prepare the customized environment where you want to train an AI.

The basic steps are as below:

* Prepare the customized environment
* Prepare a config
* Train and evaluate on the customized environment

Prepare the Customized Environment
-------------------------------------

To use the customized environment, write a wrapper class named :code:`Environment` in :code:`my_env.py` according to the HandyRL's format

.. note::
    The games HandyRL supports are as below:

    * Turn-based game (alternating game) (`tictactoe.py <https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/tictactoe.py>`_, `geister.py <https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/geister.py>`_)
    * Simultaneous game (`hungry_geese.py <https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/kaggle/hungry_geese.py>`_)

To see all interfaces of environment, check `environment.py <https://github.com/DeNA/HandyRL/blob/master/handyrl/environment.py>`_.

Turn-Based Game
--------------------------------------

Let's create a class :code:`Environment`. We recommend to check compared with the implementation of sample environment like sample game `Tic-Tac-Toe <https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/tictactoe.py>`_.

.. code-block:: python

    # my_env.py
    class Environment:
        def __init__(self, args={}):
            ...

Next, implement :code:`reset()` which resets the game. Then :code:`play()` or :code:`step()` methods should be implemented. :code:`play()` and :code:`step()` function step the game state with one step. Note that the difference between those methods is the argument.

.. code-block:: python

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
    # Should be defined in games which has simultaneous trainsition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)


:code:`terminal()` returns whether the game finished or not.

.. code-block:: python

    #
    # Should be defined in all games
    #
    def terminal(self):
        return False


:code:`players()` returns the list of player id, and :code:`turn()` returns the current player id.

.. code-block:: python

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


:code:`reward()` and :code:`outcome()` return the reward and outcome respectively at the step.

.. code-block:: python

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


:code:`action_length()` returns the length of all legal actions. :code:`legal_actions()` returns the available actions (indices) in the step. Note that the actions are represented by int.

.. code-block:: python

    #
    # Should be defined in all games
    #
    def action_length(self):
        return 4  # example: right(0), bottom(1), left(2), top(3)

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        return [0, 3]  # right(0) and top(3) available


Finally, the features to feed neural network is implemented in :code:`observation()`. This method need to return the input array for neural network.

.. code-block:: python

    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        obs = ...
        return np.array(obs)  # this array will be fed to neural network


Simultaneous Game
------------------------------

In simultaneous game, you need to implement different methods for handling multi players. We recommend to check compared with the implementation of sample environment like sample game `hungry_geese.py <https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/kaggle/hungry_geese.py>`_.

First, :code:`step()` method is required to handle the actions of multi players.

.. code-block:: python

    #
    # Should be defined in games which has simultaneous transition
    #
    def step(self, actions):
        ...


Next, :code:`turns()` returns the list of player id that can act in the step.

.. code-block:: python

    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [0, 1, 2, 3]


Prepare your config file
--------------------------

Set the name of the environment or path to environment script to :code:`env`.

.. code-block:: yaml

    env_args:
        env: 'path.to.my_env'


Train and Evaluate with Customized Environment
-----------------------------------------------------

.. code-block:: python

    python main.py --train

.. code-block:: python

    python main.py --eval models/1.pth 100 4

