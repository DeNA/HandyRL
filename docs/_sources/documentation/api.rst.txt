=================
API
=================

In HandyRL, :code:`main.py` is an entry-point for training and evaluation. This page explains the API interfaces of :code:`main.py`.

Also see the usage for more details:

* `training <../tutorial/tictactoe.html>`_
* `training (server version) <large_scale_training.html>`_


Training API
----------------------

.. code-block:: bash

    python main.py [--train|-t]


Evaluation API
----------------------

.. code-block:: bash

    python main.py [--eval|-e] MODEL_PATH NUM_GAMES NUM_PROCESSES

* :code:`MODEL_PATH`

  * model path to evaluate
* :code:`NUM_GAMES`

  * the number of games to evaluate
* :code:`NUM_PROCESSES`

  * the number of processes to evaluate model in parallel

Training Server API
-----------------------

.. code-block:: bash

    python main.py [--train-server|-ts]


Worker API
------------------------

.. code-block:: bash

    python main.py [--worker|-w]


Evaluation Server API
------------------------

.. code-block:: bash

    python main.py [--eval-server|-es] NUM_GAMES NUM_PROCESSES


* :code:`NUM_GAMES`

  * the number of games to evaluate
* :code:`NUM_PROCESSES`

  * the number of processes to evaluate model in parallel

Evaluation Client API
------------------------

.. code-block:: bash

    python main.py [--eval-client|-ec] MODEL_PATH


* :code:`MODEL_PATH`: 

  * model path to evaluate
