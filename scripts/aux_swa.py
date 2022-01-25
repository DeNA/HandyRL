
# Usage: python3 script/aux_swa.py [FINAL_EPOCH] [EPOCHS] [EPOCH_STEP]

import os
import sys

sys.path.append('./')

import yaml

import torch
from torch.optim.swa_utils import AveragedModel

from handyrl.environment import make_env


#
# SWA (running equal averaging)
#

model_dir = 'models'
saved_model_path = os.path.join('models', 'swa.pth')

ed, length = int(sys.argv[1]), int(sys.argv[2])
step = 1
if len(sys.argv) >= 4:
    step

model_ids = [str(i) + '.pth' for i in range(ed - length + 1, ed + 1, step)]

with open('config.yaml') as f:
    args = yaml.safe_load(f)

env = make_env(args['env_args'])
model = env.net()
model.load_state_dict(torch.load(os.path.join(model_dir, model_ids[0])), strict=True)

def _avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)

swa_model = AveragedModel(model, avg_fn=_avg_fn)

for model_id in model_ids:
    model.load_state_dict(torch.load(os.path.join(model_dir, model_id)), strict=True)
    swa_model.update_parameters(model)

torch.save(swa_model.module.state_dict(), saved_model_path)

print('Saved %s' % saved_model_path)

#
# Test (load in strict=True)
#

model = env.net()
model.load_state_dict(torch.load(saved_model_path), strict=True)
