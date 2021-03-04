## Large Scale Training

HandyRL allows you to train a model remotely on a large scale. This means that the lerner and the worker run separately on different machines. This is useful to efficiently use machine resources such as GPU and CPU because GPU becomes a bottleneck in the learner and CPU in the worker. GPU accelerates model update and many CPUs are required to step a game environment. Thus, you can train a model with a machine of 1 GPU (but small CPUs), and generate episode data with a machine of 64 CPUs (but no GPU).

Let's start distributed reinforcement learning!


### Step 1: Remote configuration

When you use remote machines as worker clients, you need to set the training server address in each client (training server = learner). 


```yaml
worker_args:
    server_address: '127.0.0.1'  # Set training server address to be connected from worker
    ...
```


NOTE: When you train a model on cloud(e.g. GCP, AWS), the internal/external IP of virtual machine can be set here.


### Step 2: Start training server

Run the following command in the training server to start distributed training. Then, the server listens to connections from workers. The trained models are saved in `models` folder.

```
python main.py --train-server
```


### Step 3: Start workers

After starting the training server, you can start the workers for self-play (episode generation) and evaluation. In HandyRL, anytime and from anywhere, (multi-node) multiple workers can connect to the server.


```
python main.py --worker
```


### Step 4: Evaluate

After training, you can evaluate the model against any models. The below code evaluate the model of epoch 1 for 100 games with 4 processes.


```
python main.py --eval models/1.pth 100 4
```