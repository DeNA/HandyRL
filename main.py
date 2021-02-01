# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import sys
import yaml


default_config = {
    'train_args': {
        'observation': False,
        'compress_steps': 4,
        'eval_rate': 0.1,
    },
}


if __name__ == '__main__':
    def recursive_update(dct, merge_dct):
        for k, v in merge_dct.items():
            if k in dct and isinstance(dct[k], dict) and isinstance(v, dict):
                recursive_update(dct[k], v)
            elif not k in dct:
                dct[k] = v
        return dct

    with open('config.yaml') as f:
        config_args = yaml.safe_load(f)
    args = default_config
    recursive_update(args, config_args)
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
