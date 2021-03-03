## Config Parameters

This page contains the description of all parameters in `config.yaml`. HandyRL uses yaml-style configuration for training and evaluation.

### Environment Parameters (env_args)

This parameters are used for training and evaluation.

* `env`, type = string
    * environment name
    * **NOTE**: only used for print in terminal for now
* `source`, type = string
    * path to environment module
    * **NOTE**: if your environment module is on `handyrl/envs/your_env.py`, set `handyrl.envs.your_env` (split `.py`)


### Training Parameters (train_args)

This parameters are used for training (`python main.py --train`, `python main.py --train-server`).


* `turn_based_training`, type = bool
    * flag for alternating turns games or not
    * set `True` for alternating turns games (e.g. Tic-Tac-Toe and Geister), `False` for simultaneous games (e.g. HungryGeese)
* `observation`, type = bool
    * Whether using opponent features in training
* `gamma`, type = double, constraints: 0.0 <= `gamma` <= 1.0
    * discount rate
* `forward_steps`, type = int
    * steps used to make n-step return estimates for calculating targets of value and advantages of policy
* `compress_steps`, type = int
    * steps to compress episode data for efficient data handling
    * **NOTE** system parameter, so basically no need to change
* `entropy_regularization`, type = double, constraints: `entropy_regularization` >= 0.0
    * coefficient of entropy regularization
* `entropy_regularization_decay`, type = double, constraints: 0.0 <= `entropy_regularization` <= 1.0
    * decay rate of entropy regularization over step progress
    * **NOTE** HandyRL reduces the effect of entropy regularization as the turn progresses
    * **NOTE** larger value decrease the effect, smaller value increase the effect
* `update_episodes`, type = int
    * the interval number of episode to update and save model
    * the models in workers are updated in this timing
* `batch_size`, type = int
    * batch size
* `minimum_episodes`, type = int
    * minimum buffer size to store episode data
    * training starts after episode data stored more than `minimum_episodes`
* `maximum_episodes`, type = int, constraints: `maximum_episodes` >= `minimum_episodes`
    * maximum buffer size to store episode data
    * exceeded episode is popped from oldest one
* `num_batchers`, type = int
    * the number of batcher that makes batch data in multi-process
* `eval_rate`, type = double, constraints: 0.0 <= `eval_rate` <= 1.0
    * ratio of evaluation worker, the rest is the workers of data generation (self-play)
* `worker`
    * `num_parallel`, type = int
        * the number of worker processes
        * `num_parallel` workers are generated automatically for data generation (self-play) and evaluation
* `lambda`, type = double, constraints: 0.0 <= `lambda` <= 1.0
    * the parameter for lambda values
    * **NOTE** HandyRL computes values using in lambda fashion such as TD, V-Trace, UPGO
* `policy_target`, type = enum
    * advantage for policy gradient loss
    * `MC`, monte carlo
    * `TD`, TD(λ)
    * `VTRACE`, [V-Trace described in IMPALA paper](https://arxiv.org/abs/1802.01561)
    * `UPGO`, [UPGO described in AlphaStar paper](https://www.nature.com/articles/s41586-019-1724-z)
* `value_target`, type = enum
    * value target for value loss
    * `MC`, monte carlo
    * `TD`, TD(λ)
    * `VTRACE`, [V-Trace described in IMPALA paper](https://arxiv.org/abs/1802.01561)
    * `UPGO`, [UPGO described in AlphaStar paper](https://www.nature.com/articles/s41586-019-1724-z)
* `seed`, type = int
    * used to set seed in learner and actor
    * **NOTE** this seed cannot guarantee the reproducibility for now
* `restart_epoch`, type = int
    * number of epochs to restart training
    * when setting `restart_epoch = 100`, the training restarts from `models/100.pth` and the next model is saved from `models/101.pth`
    * **NOTE** HandyRL check `models` directory

### Worker Parameters (worker_args)

This parameters are used only for worker of distributed training (`python main.py --worker`).

* `server_address`, type = string
    * training server address to be connected from worker
    * **NOTE**: when training a model on cloud (e.g. GCP, AWS), the internal/external IP of virtual machine can be set here
* `num_parallel`, type = int
    * the number of worker processes
    * `num_parallel` workers are generated automatically for data generation (self-play) and evaluation
