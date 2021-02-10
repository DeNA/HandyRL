# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import os
import resource
import sys
import yaml


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, resource.RLIM_INFINITY))

    with open('config.yaml') as f:
        args = yaml.safe_load(f)
    print(args)

    if len(sys.argv) < 2:
        print('Please set mode of HandyRL.')
        exit(1)

    mode = sys.argv[1]

    if mode == '--train' or mode == '-t':
        from handyrl.train import train_main as main
        main(args)
    if mode == '--train-server' or mode == '-ts':
        from handyrl.train import train_server_main as main
        main(args)
    elif mode == '--worker' or mode == '-w':
        from handyrl.worker import worker_main as main
        main(args)
    elif mode == '--eval' or mode == '-e':
        from handyrl.evaluation import eval_main as main
        main(args, sys.argv[2:])
    elif mode == '--eval-server' or mode == '-es':
        from handyrl.evaluation import eval_server_main as main
        main(args, sys.argv[2:])
    elif mode == '--eval-client' or mode == '-ec':
        from handyrl.evaluation import eval_client_main as main
        main(args, sys.argv[2:])
    else:
        print('Not found mode %s.' % mode)
