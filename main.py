# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import sys
import yaml

from handyrl.train import train_main, train_server_main
from handyrl.worker import worker_main
from handyrl.evaluation import eval_main, eval_server_main, eval_client_main


if __name__ == '__main__':
    with open('config.yaml') as f:
        args = yaml.safe_load(f)
    print(args)

    if len(sys.argv) < 2:
        print('Please set mode of HandyRL.')
        exit(1)

    mode = sys.argv[1]

    if mode == '--train' or mode == '-t':
        train_main(args)
    if mode == '--train-server' or mode == '-ts':
        train_server_main(args)
    elif mode == '--worker' or mode == '-w':
        worker_main(args)
    elif mode == '--eval' or mode == '-e':
        eval_main(args, sys.argv[2:])
    elif mode == '--eval-server' or mode == '-es':
        eval_server_main(args, sys.argv[2:])
    elif mode == '--eval-client' or mode == '-ec':
        eval_client_main(args, sys.argv[2:])
    else:
        print('Not found mode %s.' % mode)
