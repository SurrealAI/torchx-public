import torch.nn.modules.dropout as _builtin
from .base import get_torch_builtin_modules
from .container import Lambda


def _generate_code():
    "generate code for this module"
    pkg = 'dropout'
    print('#', '='*25, 'generated', '='*25)
    print('_wrap = Lambda.wrap_same_shape_class\n')
    for cls_name in get_torch_builtin_modules(pkg):
        print(cls_name, '=', "_wrap(_builtin.{0})".format(cls_name))

# ========================= generated =========================
_wrap = Lambda.wrap_same_shape_class

AlphaDropout = _wrap(_builtin.AlphaDropout)
Dropout = _wrap(_builtin.Dropout)
Dropout2d = _wrap(_builtin.Dropout2d)
Dropout3d = _wrap(_builtin.Dropout3d)
