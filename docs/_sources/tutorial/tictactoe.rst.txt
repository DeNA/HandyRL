==============================
Training for Tic-Tac-Toe
==============================

This section shows the training a model for `Tic-Tac-Toe <https://en.wikipedia.org/wiki/Tic-tac-toe>`_. Tic-Tac-Toe is a very simple game. You can play by googling "Tic-Tac-Toe".

Step 1: Set up configuration
------------------------------

Set :code:`config.yaml` for your training configuration. When you run a training with Tic-Tac-Toe and batch size 64, set like the following:


.. code-block:: yaml

    env_args:
        env: 'TicTacToe'

    train_args:
        ...
        batch_size: 64
        ...


.. note:: `Here is the list of default games implemented in HandyRL <https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/>`_. All parameters are shown in `Config Parameters <../documentation/parameters.html>`_.


Step 2: Train!
--------------------------------

After creating the configuration, you can start training by running the following command. The trained models are saved in :code:`models` folder every :code:`update_episodes` described in :code:`config.yaml`.

.. code-block:: bash

    python main.py --train


Step 3: Evaluate
--------------------------------

After training, you can evaluate the model against any models. The below code evaluate the model of epoch 1 for 100 games with 4 processes.


.. code-block:: bash

    python main.py --eval models/1.pth 100 4


.. note:: The default opponent AI is the random agent implemented in :code:`evaluation.py`. You can change the agent with any of your agents.
