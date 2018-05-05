"""
Hack torch.autograd.Variable:
from surreal.utils.pytorch import GpuVariable as Variable
"""
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torchx.device import get_torchx_device_dtype
from torchx.utils.common import SaveInitArgs
from torchx.nn.compute import *


class Module(nn.Module, SaveInitArgs):
    """
    All models in Surreal should extend this module, not pytorch one
    """
    def __call__(self, *args, **kwargs):
        """
        transfer to the device in torchx scope before forward pass
        """
        devices, dtype = get_torchx_device_dtype()
        self.to(device=devices[0], dtype=dtype)
        return super().__call__(*args, **kwargs)

    def copy_from(self, other_module):
        th_copy_module(other_module, self)
        return self

    def copy_to(self, other_module):
        th_copy_module(self, other_module)
        return self

    def freeze(self):
        "same effect as with torch.no_grad()"
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        "same effect as with torch.enable_grad()"
        for param in self.parameters():
            param.requires_grad = True

    def soft_update(self, other_module, tau):
        th_soft_update(target=self, source=other_module, tau=tau)

    def hard_update(self, other_module, tau):
        th_hard_update(target=self, source=other_module)

    def clip_grad_value(self, max_value):
        with torch.no_grad():
            nn.utils.clip_grad_value_(self.parameters(), max_value)

    def clip_grad_norm(self, max_norm, norm_type=2):
        """
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        with torch.no_grad():
            return nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=max_norm,
                norm_type=norm_type
            )

    def save(self, fname):
        save_dict = OrderedDict()
        # from meta class SaveInitArgs
        save_dict['init_args_dict'] = self.init_args_dict
        save_dict['torch'] = self.state_dict()
        torch.save(save_dict, fname)

    def load(self, fname):
        save_dict = torch.load(os.path.expanduser(fname))
        self.load_state_dict(save_dict['torch'])

    @classmethod
    def load_with_init(cls, fname):
        """
        Also load the saved constructor arguments
        """
        save_dict = torch.load(os.path.expanduser(fname))
        init_args_dict = save_dict['init_args_dict']
        positionals = init_args_dict.pop(None, [])  # under key None
        kwargs = init_args_dict
        net = cls(*positionals, **kwargs)
        net.load_state_dict(save_dict['torch'])
        return net

    def parameters_to_binary(self):
        flattened = th_flatten_tensors(self)
        return flattened.cpu().numpy().tostring()

    def parameters_from_binary(self, binary):
        """
        Assumes np.float32
        """
        buffer = np.fromstring(binary, dtype=np.float32)
        buffer = torch.from_numpy(buffer)
        new_params = th_unflatten_tensors(buffer, self)
        with torch.no_grad():
            for p, n in zip(self.parameters(), new_params):
                p.copy_(n)

    def clone(self):
        if self.init_args_dict is None:
            raise ValueError('init_args_dict not saved. '
             'This happens when the layer __init__() accepts special things like *args.'
             ' Please manually instantiate the object and use self.load() instead')
        return type(self)(**self.init_args_dict).copy_from(self)

