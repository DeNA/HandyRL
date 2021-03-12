======================
Config Parameters
======================

This page contains the description of all parameters in :code:`config.yaml`. HandyRL uses yaml-style configuration for training and evaluation.

Environment Parameters (env_args)
-------------------------------------

This parameters are used for training and evaluation.

* :code:`env`, type = string

  * environment name
  * **NOTE**: Default games: TicTacToe, Geister, ParallelTicTacToe, HungryGeese
  * **NOTE**: if your environment module is on :code:`handyrl/envs/your_env.py`, set :code:`handyrl.envs.your_env` (split :code:`.py`)


Training Parameters (train_args)
---------------------------------------

This parameters are used for training (:code:`python main.py --train`, :code:`python main.py --train-server`).


* :code:`turn_based_training`, type = bool

  * flag for alternating turns games or not
  * set :code:`True` for alternating turns games (e.g. Tic-Tac-Toe and Geister), :code:`False` for simultaneous games (e.g. HungryGeese)
* :code:`observation`, type = bool

  * Whether using opponent features in training
* :code:`gamma`, type = double, constraints: 0.0 <= :code:`gamma` <= 1.0

  * discount rate
* :code:`forward_steps`, type = int

  * steps used to make n-step return estimates for calculating targets of value and advantages of policy
* :code:`compress_steps`, type = int

  * steps to compress episode data for efficient data handling
  * **NOTE** system parameter, so basically no need to change
* :code:`entropy_regularization`, type = double, constraints: :code:`entropy_regularization` >= 0.0

  * coefficient of entropy regularization
* :code:`entropy_regularization_decay`, type = double, constraints: 0.0 <= :code:`entropy_regularization` <= 1.0

  * decay rate of entropy regularization over step progress
  * **NOTE** HandyRL reduces the effect of entropy regularization as the turn progresses
  * **NOTE** larger value decrease the effect, smaller value increase the effect
* :code:`update_episodes`, type = int

  * the interval number of episode to update and save model
  * the models in workers are updated in this timing
* :code:`batch_size`, type = int

  * batch size
* :code:`minimum_episodes`, type = int

  * minimum buffer size to store episode data
  * training starts after episode data stored more than :code:`minimum_episodes`
* :code:`maximum_episodes`, type = int, constraints: :code:`maximum_episodes` >= :code:`minimum_episodes`

  * maximum buffer size to store episode data
  * exceeded episode is popped from oldest one
* :code:`num_batchers`, type = int

  * the number of batcher that makes batch data in multi-process
* :code:`eval_rate`, type = double, constraints: 0.0 <= :code:`eval_rate` <= 1.0

  * ratio of evaluation worker, the rest is the workers of data generation (self-play)
* :code:`worker`

  * :code:`num_parallel`, type = int

    * the number of worker processes
    * :code:`num_parallel` workers are generated automatically for data generation (self-play) and evaluation
* :code:`lambda`, type = double, constraints: 0.0 <= :code:`lambda` <= 1.0

  * the parameter for lambda values
  * **NOTE** HandyRL computes values using in lambda fashion such as TD, V-Trace, UPGO
* :code:`policy_target`, type = enum

  * advantage for policy gradient loss
  * :code:`MC`, monte carlo
  * :code:`TD`, TD(λ)
  * :code:`VTRACE`, `V-Trace described in IMPALA paper <https://arxiv.org/abs/1802.01561>`_
  * :code:`UPGO`, `UPGO described in AlphaStar paper <https://www.nature.com/articles/s41586-019-1724-z>`_
* :code:`value_target`, type = enum

  * value target for value loss
  * :code:`MC`, monte carlo
  * :code:`TD`, TD(λ)
  * :code:`VTRACE`, `V-Trace described in IMPALA paper <https://arxiv.org/abs/1802.01561>`_
  * :code:`UPGO`, `UPGO described in AlphaStar paper <https://www.nature.com/articles/s41586-019-1724-z>`_
* :code:`seed`, type = int

  * used to set seed in learner and actor
  * **NOTE** this seed cannot guarantee the reproducibility for now
* :code:`restart_epoch`, type = int

  * number of epochs to restart training
  * when setting :code:`restart_epoch = 100`, the training restarts from :code:`models/100.pth` and the next model is saved from :code:`models/101.pth`
  * **NOTE** HandyRL check :code:`models` directory


Worker Parameters (worker_args)
---------------------------------

This parameters are used only for worker of distributed training (:code:`python main.py --worker`).

* :code:`server_address`, type = string

  * training server address to be connected from worker
  * **NOTE**: when training a model on cloud (e.g. GCP, AWS), the internal/external IP of virtual machine can be set here
* :code:`num_parallel`, type = int

  * the number of worker processes
  * :code:`num_parallel` workers are generated automatically for data generation (self-play) and evaluation
