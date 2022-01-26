
# Usage: python3 script/make_onnx_model.py MODEL_PATH

import sys
import yaml
import torch

sys.path.append('./')

from handyrl.environment import make_env
from handyrl.model import to_torch
from handyrl.util import map_r


model_path = sys.argv[-1]
saved_model_path = model_path + '.onnx'

with open('config.yaml') as f:
    args = yaml.safe_load(f)

env = make_env(args['env_args'])
model = env.net()
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()
print('loaded PyTorch model from %s' % model_path)

env.reset()
obs = to_torch(env.observation(player=env.turn()))
obs = map_r(obs, lambda x: x.unsqueeze(0))

hidden = model.init_hidden([1]) if hasattr(model, 'init_hidden') else None
inputs = obs, hidden

# You can specify meaningful names for the inputs here.
input_names = []
map_r(obs, lambda y: input_names.append('input.' + str(len(input_names))))

hidden_names = []
if hidden is not None:
    map_r(hidden, lambda y: hidden_names.append('hidden.' + str(len(hidden_names))))
    input_names += hidden_names

outputs = model(*inputs)
output_names = list(outputs.keys())
if 'hidden' in output_names:
     index = output_names.index('hidden')
     output_names = output_names[:index] + [name + 'o' for name in hidden_names] + output_names[index+1:]

print('input =', input_names)
print('output =', output_names)

dynamic_axes = {name: {0: 'batch_size'} for name in (input_names + output_names)}

torch.onnx.export(model, inputs, saved_model_path,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes)
print('saved ONNX model to %s' % saved_model_path)
