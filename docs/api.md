## API

In HandyRL, `main.py` is an entry-point for training and evaluation. This page explains the API interfaces of `main.py`.

Also see the usage for more details:
* [training](../README.md#train-ai-model-for-tic-tac-toe)
* [training (server version)](large_scale_training.md)


### Training API

```
python main.py [--train|-t]
```

### Evaluation API

```
python main.py [--eval|-e] MODEL_PATH NUM_GAMES NUM_PROCESSES
```

* `MODEL_PATH`
    * model path to evaluate
* `NUM_GAMES`
    * the number of games to evaluate
* `NUM_PROCESSES`
    * the number of processes to evaluate model in parallel

### Training Server API

```
python main.py [--train-server|-ts]
```

### Worker API

```
python main.py [--worker|-w]
```

### Evaluation Server API

```
python main.py [--eval-server|-es] NUM_GAMES NUM_PROCESSES
```

* `NUM_GAMES`
    * the number of games to evaluate
* `NUM_PROCESSES`
    * the number of processes to evaluate model in parallel

### Evaluation Client API

```
python main.py [--eval-client|-ec] MODEL_PATH
```

* `MODEL_PATH`
    * model path to evaluate
