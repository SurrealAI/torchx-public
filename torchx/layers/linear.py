import torch
from .base import Layer


class Linear(Layer):
    def __init__(self, out_features, bias=True):
        super().__init__()
        self.out_features = out_features
        self._has_bias = bias
        self._fc = None

    def _build(self, input_shape):
        in_features = input_shape[-1]
        self._fc = torch.nn.Linear(
            in_features, self.out_features, bias=self._has_bias
        )

    def forward(self, x):
        return self._fc(x)

    def get_output_shape(self, input_shape):
        return (*input_shape[:-1], self.out_features)

    def native_module(self):
        return self._fc


class Dense(Linear):
    "Alias for Linear"
    pass


# def fc_layers(input_size, output_size, hiddens, initializer='xavier'):
#     assert isinstance(hiddens, (list, tuple))
#     fcs = nn.ModuleList() # IMPORTANT for .cuda() to work!!
#     layers = [input_size] + hiddens + [output_size]
#     for prev, next in zip(layers[:-1], layers[1:]):
#         fcs.append(nn.Linear(prev, next))
#     if initializer == 'xavier':
#         U.torch_init_module(fcs)
#     return fcs
#
#
# # TODO MLP rewrite as layer
# class MLP(U.Module):
#     def __init__(self, input_size, output_size, hiddens, activation=None):
#         super().__init__()
#         if activation is None:
#             self.activation = F.relu
#         else:
#             raise NotImplementedError # TODO: other activators
#         self.layers = fc_layers(input_size=input_size,
#                                 output_size=output_size,
#                                 hiddens=hiddens)
#
#     def reinitialize(self):
#         U.torch_init_module(self.layers)
#
#     def forward(self, x):
#         for is_last, fc in U.iter_last(self.layers):
#             x = fc(x)
#             if not is_last:
#                 x = self.activation(x)
#         return x
