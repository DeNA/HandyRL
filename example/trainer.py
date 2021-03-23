import os
import sys
import yaml

sys.path.append('../')

from handyrl import Learner, WorkerCluster
from handyrl.envs.tictactoe import Environment
from handyrl.envs.tictactoe import SimpleConv2dModel as Net


if __name__ == '__main__':
    with open('../config.yaml') as f:
        args = yaml.safe_load(f)
    print(args)

    learner = Learner(env=Environment, net=Net, args=args)
    learner.run()
