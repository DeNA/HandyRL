# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import os
os.environ['OMP_NUM_THREADS'] = '1'

from .train import Learner
