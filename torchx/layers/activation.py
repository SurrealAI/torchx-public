import torch.nn.modules.activation as _builtin
from .base import get_torch_builtin_modules, Layer
from .container import Lambda


def _generate_code():
    "generate code for this module"
    pkg = 'activation'
    print('#', '='*25, 'generated', '='*25)
    print('_wrap = Lambda.wrap_same_shape_class\n')
    for cls_name in get_torch_builtin_modules(pkg):
        if cls_name in ['PReLU']:
            continue
        print(cls_name, '=', "_wrap(_builtin.{0})".format(cls_name))


# PReLU has learnable parameters that change with upsteam shape
class PReLU(Layer):
    def __init__(self, shared=True, init=0.25):
        """
        https://pytorch.org/docs/stable/nn.html#prelu

        Args:
            shared: True to have only one learnable parameter shared across all
                channels; False to have a different parameter for each channel
        """
        super().__init__()
        self._shared = shared
        self._activation = None
        self._kwargs = {'init': init}

    def _build(self, input_shape):
        if self._shared:
            num_parameters = 1
        else:
            num_parameters = input_shape[1]  # upstream channel
        self._activation = _builtin.PReLU(num_parameters, **self._kwargs)

    def forward(self, x):
        return self._activation(x)

    def get_output_shape(self, input_shape):
        return input_shape


# ========================= generated =========================
_wrap = Lambda.wrap_same_shape_class

ELU = _wrap(_builtin.ELU)
GLU = _wrap(_builtin.GLU)
Hardshrink = _wrap(_builtin.Hardshrink)
Hardtanh = _wrap(_builtin.Hardtanh)
LeakyReLU = _wrap(_builtin.LeakyReLU)
LogSigmoid = _wrap(_builtin.LogSigmoid)
LogSoftmax = _wrap(_builtin.LogSoftmax)
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
