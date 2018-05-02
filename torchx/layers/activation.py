import torch.nn.modules.activation as _builtin
from .base import get_torch_builtin_modules
from .container import Lambda


def _generate_code():
    "generate code for this module"
    pkg = 'activation'
    print('#', '='*25, 'generated', '='*25)
    print('_wrap = Lambda.wrap_same_shape_class\n')
    for cls_name in get_torch_builtin_modules(pkg):
        print(cls_name, '=', "_wrap(_builtin.{0})".format(cls_name))


# ========================= generated =========================
_wrap = Lambda.wrap_same_shape_class

ELU = _wrap(_builtin.ELU)
GLU = _wrap(_builtin.GLU)
Hardshrink = _wrap(_builtin.Hardshrink)
Hardtanh = _wrap(_builtin.Hardtanh)
LeakyReLU = _wrap(_builtin.LeakyReLU)
LogSigmoid = _wrap(_builtin.LogSigmoid)
LogSoftmax = _wrap(_builtin.LogSoftmax)
PReLU = _wrap(_builtin.PReLU)
RReLU = _wrap(_builtin.RReLU)
ReLU = _wrap(_builtin.ReLU)
ReLU6 = _wrap(_builtin.ReLU6)
SELU = _wrap(_builtin.SELU)
Sigmoid = _wrap(_builtin.Sigmoid)
Softmax = _wrap(_builtin.Softmax)
Softmax2d = _wrap(_builtin.Softmax2d)
Softmin = _wrap(_builtin.Softmin)
Softplus = _wrap(_builtin.Softplus)
Softshrink = _wrap(_builtin.Softshrink)
Softsign = _wrap(_builtin.Softsign)
Tanh = _wrap(_builtin.Tanh)
Tanhshrink = _wrap(_builtin.Tanhshrink)
Threshold = _wrap(_builtin.Threshold)
