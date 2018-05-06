import torch
import torch.nn as nn
import torch.nn.functional as F

import torchx as tx
import torchx.utils as U
from torchx.layers import *
import torchx.nn as nnx


# pprint(nnx.th_all_nn_modules_names())
# pprint(nnx.th_all_initializers_names())

T = torch.zeros((3,5))
print(nnx.th_initializer('kaiming_uniform')(T))
print(nnx.th_initializer('zero')(T))
print(nnx.th_initializer({
    'method': 'normal',
    'std': 20,
    'mean': 200
})(T))

print('Conv2d init')
m = nn.Conv2d(5, 7, (3,3))
nnx.th_initialize_module(
    m,
    {'method': 'normal', 'mean': 23},
    {'method': 'constant', 'val': -4}
)
print(m.weight.size(), U.to_scalar(m.weight.mean()))
print(m.bias.size(), U.to_scalar(m.bias.mean()))

print('LSTM init')
m = nn.LSTM(5, 7, num_layers=4, batch_first=True, bidirectional=True)
nnx.th_initialize_module(
    m,
    ({'method': 'normal', 'mean': 30}, {'method':'constant', 'val': 46}),
    ('zero', {'method': 'normal', 'mean': -30}),
)

for attr in sorted(dir(m)):
    if attr.startswith('weight_') or attr.startswith('bias_'):
        param = getattr(m, attr)
        print(attr, param.size(), U.to_scalar(param.mean()))

