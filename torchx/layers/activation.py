import torch.nn.modules.activation as _builtin
from .base import get_torch_builtin_modules
from .container import Lambda


def _generate_code():
    "generate code for this module"
    pkg = 'activation'
    print('#', '='*25, 'generated', '='*25)
    print('_wrap = Lambda.wrap_same_shape_class\n')
    for cls_name in get_torch_builtin_modules(pkg):
        print(cls_name, '=', "_wrap(_builtin.{0}, '{0}')".format(cls_name))


# ========================= generated =========================
_wrap = Lambda.wrap_same_shape_class

ELU = _wrap(_builtin.ELU, 'ELU')
GLU = _wrap(_builtin.GLU, 'GLU')
Hardshrink = _wrap(_builtin.Hardshrink, 'Hardshrink')
Hardtanh = _wrap(_builtin.Hardtanh, 'Hardtanh')
LeakyReLU = _wrap(_builtin.LeakyReLU, 'LeakyReLU')
LogSigmoid = _wrap(_builtin.LogSigmoid, 'LogSigmoid')
LogSoftmax = _wrap(_builtin.LogSoftmax, 'LogSoftmax')
PReLU = _wrap(_builtin.PReLU, 'PReLU')
RReLU = _wrap(_builtin.RReLU, 'RReLU')
ReLU = _wrap(_builtin.ReLU, 'ReLU')
ReLU6 = _wrap(_builtin.ReLU6, 'ReLU6')
SELU = _wrap(_builtin.SELU, 'SELU')
Sigmoid = _wrap(_builtin.Sigmoid, 'Sigmoid')
Softmax = _wrap(_builtin.Softmax, 'Softmax')
Softmax2d = _wrap(_builtin.Softmax2d, 'Softmax2d')
Softmin = _wrap(_builtin.Softmin, 'Softmin')
Softplus = _wrap(_builtin.Softplus, 'Softplus')
Softshrink = _wrap(_builtin.Softshrink, 'Softshrink')
Softsign = _wrap(_builtin.Softsign, 'Softsign')
Tanh = _wrap(_builtin.Tanh, 'Tanh')
Tanhshrink = _wrap(_builtin.Tanhshrink, 'Tanhshrink')
Threshold = _wrap(_builtin.Threshold, 'Threshold')

