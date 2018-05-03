import torch
from copy import deepcopy
import inspect
import torch.nn as nn


def th_all_nn_modules_dict():
    """
    Returns:
      dict of {'ModuleName': ModuleClass} in torch.nn
    """
    return {
        cls_name: cls for cls_name, cls
        in vars(nn).items()
        if not cls_name.startswith('_')
            and inspect.isclass(cls)
    }


def th_all_nn_modules_names():
    """
    Returns:
      list of all nn module names, sorted alphabetically
    """
    return sorted(list(th_all_nn_modules_dict().keys()))


def is_conv_module(module, dim=None):
    if dim is None:
        cls = nn.modules.conv._ConvNd
    elif dim == 1:
        cls = nn.Conv1d
    elif dim == 2:
        cls = nn.Conv2d
    elif dim == 3:
        cls = nn.Conv3d
    else:
        raise ValueError('unsupported dim', dim)
    return isinstance(module, cls)


def th_all_initializer_dict():
    """
    In v0.4, all initializers end with underscore to denote in-place
    the returned initializer name does NOT have the trailing underscore

    Returns:
      dict of {'InitializerName': initializer_method} in torch.nn.init
    """
    initers = {
        init_name[:-1]: init_method for init_name, init_method
        in vars(nn.init).items()
        if not init_name.startswith('_')
            and init_name.endswith('_')
            and callable(init_method)
    }
    non_builtin = {
        'zero': lambda tensor: nn.init.constant_(tensor, 0)
    }
    initers.update(non_builtin)
    return initers


def th_all_initializer_names():
    """
    Returns:
      list of initializer names, sorted alphabetically
    """
    return sorted(list(th_all_initializer_dict().keys()))


def th_initializer(spec):
    """
    Args:
      spec: either of 2 formats
        - initializer name: apply init function with default args
        - dict with {'method': 'initializer name', **kwargs} apply init function
          with custom kwargs, please check PyTorch API
          http://pytorch.org/docs/master/_modules/torch/nn/init.html
    """
    all_initers = th_all_initializer_dict()
    if isinstance(spec, str):
        spec = spec.strip('_')
        assert spec in all_initers, \
            ('valid initializers are:', th_all_initializer_names())
        return all_initers[spec]
    else:
        spec = deepcopy(spec)
        assert 'method' in spec
        method = spec.pop('method')
        assert method in all_initers, \
            ('valid initializers are:', th_all_initializer_names())
        method = all_initers[method]
        return lambda tensor: method(tensor, **spec)


def _prefix_attrs(module, prefix):
    "th_initialize_module helper"
    attrs = [name for name in dir(module) if name.startswith(prefix)]
    assert attrs, 'module must have Parameters with prefix "{}"'.format(prefix)
    return attrs


def th_initialize_module(module,
                         weight_init='xavier_uniform',
                         bias_init='zero'):
    """
    Supports Linear, ConvNd, ConvTransposeNd, RNNs.
    Initializer spec please see `th_initializer()`
    For RNN modules, assume Parameter name start with `weight_ih_l`,
      `weight_hh_l`, `bias_ih_l` and `bias_hh_l`
      e.g. weight_ih_l3, bias_hh_l0, weight_ih_l2_reverse (for bidrectional)
      see `nn.RNNBase` source code
    For all non-RNN modules, assume Parameters `weight` and `bias`

    Args:
      module: torch module object
      weight_init: for RNN modules, you can specify a 2-tuple for different
        `weight_ih_l` and `weight_hh_l` initializers
      bias_init: for RNN modules, you can specify a 2-tuple for different
        `bias_ih_l` and `bias_hh_l` initializers
    """
    if isinstance(module, nn.RNNBase):
        if isinstance(weight_init, (tuple, list)):
            assert len(weight_init) == 2, \
                '2-tuple for different `weight_ih_l<n>` and `weight_hh_l<n>` initializers'
        else:
            weight_init = (weight_init,) * 2
        weight_init_ih = th_initializer(weight_init[0])
        weight_init_hh = th_initializer(weight_init[1])
        for w in _prefix_attrs(module, 'weight_ih_l'):
            weight_init_ih(getattr(module, w))
        for w in _prefix_attrs(module, 'weight_hh_l'):
            weight_init_hh(getattr(module, w))

        if isinstance(bias_init, (tuple, list)):
            assert len(bias_init) == 2, \
                '2-tuple for different `bias_ih_l<n>` and `bias_hh_l<n>` initializers'
        else:
            bias_init = (bias_init,) * 2
        bias_init_ih = th_initializer(bias_init[0])
        bias_init_hh = th_initializer(bias_init[1])
        for b in _prefix_attrs(module, 'bias_ih_l'):
            bias_init_ih(getattr(module, b))
        for b in _prefix_attrs(module, 'bias_hh_l'):
            bias_init_hh(getattr(module, b))
    else:  # Linear, ConvNd, etc.
        required_attrs = ['weight', 'bias']
        for attr in required_attrs:
            assert hasattr(module, attr), \
                ('module must have parameters:', required_attrs)
        weight_init = th_initializer(weight_init)
        weight_init(module.weight)
        bias_init = th_initializer(bias_init)
        bias_init(module.bias)

