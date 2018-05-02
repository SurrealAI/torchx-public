import torch
from .base import Layer


class Dense(Layer):
    def __init__(self, out_features, *, input_shape=None, bias=True):
        super().__init__(input_shape=input_shape, bias=bias)
        self.out_features = out_features

    def _build(self, input_shape):
        in_features = input_shape[-1]
        self.fc = torch.nn.Linear(in_features, self.out_features,
                                  **self.init_kwargs)

    def forward(self, x):
        return self.fc(x)

    def get_output_shape(self, input_shape):
        return (*input_shape[:-1], self.out_features)

    def __repr__(self):
        if self.is_built:
            input_features = self.input_shape[-1]
        else:
            input_features = None
        return '{}({}->{})'.format(
            self.__class__.__name__, input_features, self.out_features
        )


class Linear(Dense):
    "alias"
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
