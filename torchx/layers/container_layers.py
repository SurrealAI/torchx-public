import torch.nn as nn

from .core import Layer, SameShapeAdapter
from .placeholder import PlaceholderStruct
import torchx.utils as U


class Sequential(Layer):
    def __init__(self, layer_list, *, input_shape=None):
        """
        If input_shape is None, will inherit the input_shape from the
        first layer in `layer_list`
        """
        assert len(layer_list) >= 1
        self.layer_list = list(map(SameShapeAdapter, layer_list))
        if input_shape is None:
            input_shape = layer_list[0].input_shape
        super().__init__(input_shape=input_shape)
        # ModuleList must be created after init, otherwise PyTorch error
        self.module_list = nn.ModuleList()

    def _add_after_build(self, layer):
        # infer shape after the last added layer
        input_shape = self.get_output_shape(self.input_shape)
        layer.build(input_shape)

    def add(self, layers):
        """
        Args:
            layer: can be a single Layer or a list of Layers
        """
        if isinstance(layers, (list, tuple)):
            for layer in layers:
                self.add(layer)
            return

        layer = SameShapeAdapter(layers)
        if self.is_built:
            assert self.input_shape is not None, 'internal error'
            self._add_after_build(layer)
            self.module_list.append(layer)
        self.layer_list.append(layer)

    def _build(self, input_shape):
        for layer in self.layer_list:
            layer.build(input_shape)
            input_shape = layer.get_output_shape(input_shape)
            self.module_list.append(layer)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

    def get_output_shape(self, input_shape):
        for layer in self.layer_list:
            input_shape = layer.get_output_shape(input_shape)
        return input_shape


class TimeDistributed(Sequential):
    """
    Applies a sequence of layers to every temporal slice of an input.
    The input should be at least 3D, and the dimension of index one will be
    considered to be the temporal dimension.
    Essentially collapses dim 0 and 1, apply sequence of layers,
    and then restore to the old dim 0 and 1

    Similar to Keras API
    https://keras.io/layers/wrappers/
    """
    def __init__(self, layer_list, *, input_shape=None):
        super().__init__(layer_list, input_shape=input_shape)

    def _collapsed_shape(self, shape):
        """
        Collapse the first 2 dims: batch_size and seq_len dim
        """
        batch_size, seq_len, *remains = shape
        return (batch_size * seq_len, *remains)

    def _build(self, input_shape):
        assert len(input_shape) >= 3, \
            'TimeDistributed input_shape must be at least 3D (with batch dim)'
        super()._build(self._collapsed_shape(input_shape))

    def _add_after_build(self, layer):
        # infer shape after the last added layer
        input_shape = self.get_output_shape(self.input_shape)
        layer.build(self._collapsed_shape(input_shape))

    def forward(self, x):
        assert len(x.size()) >= 3, 'TimeDistributed input must be at least 3D'
        batch_size, seq_len, *remains = x.size()
        x = x.contiguous().view(batch_size * seq_len, *remains)
        x = super().forward(x)
        collapsed_size, *new_remains = x.size()
        assert collapsed_size == batch_size * seq_len, 'internal error'
        return x.contiguous().view(batch_size, seq_len, *new_remains)

    def get_output_shape(self, input_shape):
        assert U.is_simple_shape(input_shape)
        assert len(input_shape) >= 3, \
            'TimeDistributed input_shape must be at least 3D (with batch dim)'
        batch_size, seq_len, *remains = input_shape
        output_shape = super().get_output_shape(
            self._collapsed_shape(input_shape)
        )
        assert output_shape[0] == batch_size * seq_len, \
            'TimeDistributed shape error: collapsed dim should equal batch_size* seq_len'
        return (batch_size, seq_len) + tuple(output_shape[1:])


class Functional(Layer):
    def __init__(self, inputs, outputs):
        """
        If input_shape is None, will inherit the input_shape from the
        first layer in `layer_list`
        """
        super().__init__(input_shape=None)
        # ModuleList must be created after init, otherwise PyTorch error
        self.module_list = nn.ModuleList()
        self.inputs = PlaceholderStruct(inputs)
        self.outputs = PlaceholderStruct(outputs)
        self._input_ids = set(id(p) for p in self.inputs.flatten())

    def postorder_traverse(self):
        """
        Returns:
          list of [(layer, node_index)], postorder traversal, the next layer is
          guaranteed to have all the input placeholders computed by the
          previous layers.
        """
        ordered_layers = []
        visited_layer_ids = set()  # (id(layer), node_index)
        self._postorder_traverse_helper(
            self.outputs, ordered_layers, visited_layer_ids
        )
        return ordered_layers

    def _not_input(self, placeholder):
        """
        Check if a placeholder is not one of the inputs
        Stop graph traversal if we are at the input placeholder
        """
        return id(placeholder) not in self._input_ids

    def _postorder_traverse_helper(self, outputs,
                                   ordered_layers,
                                   visited_layer_ids):
        flattened_outputs = outputs.flatten()
        for output in flattened_outputs:
            inbound = output.inbound_layer  # None if no ancestor
            # stop traversal when encountering input placeholder
            if inbound and self._not_input(output):
                self._postorder_traverse_helper(
                    inbound.input_placeholder_nodes[output.node_index],
                    ordered_layers,
                    visited_layer_ids
                )
        for output in flattened_outputs:
            inbound = output.inbound_layer
            hashkey = (id(inbound), output.node_index)
            if (inbound and
                    self._not_input(output) and
                    hashkey not in visited_layer_ids):
                visited_layer_ids.add(hashkey)
                ordered_layers.append((inbound, output.node_index))

    def forward(self, *args, **kwargs):
        input_tensors = self._pack_call_args(args, kwargs)
        self.inputs.bind_tensors(input_tensors)
        output_tensors = None
        for layer, node_index in self.postorder_traverse():
            print('DEBUG forward layer', layer, 'node', node_index)
            input_struct = layer.input_placeholder_nodes[node_index]
            assert input_struct.all_tensor_bound(), \
                ('placeholder without bound tensor in', layer)
            output_tensors = layer(input_struct.to_tensors())
            output_node = layer.output_placeholder_nodes[node_index]
            print('DEBUG outpout', output_node, output_tensors.size())
            output_node.bind_tensors(output_tensors)
        return self.outputs.to_tensors()

    def compile(self):
        """
        """
        self.build(self.inputs.get_shape())
        return self

    def _build(self, input_shape):
        for layer, node_index in self.postorder_traverse():
            if node_index == 0:
                layer.build()
                self.module_list.append(layer)
        print('DEBUG all modules', self.module_list)

    def get_output_shape(self, input_shape):
        return self.outputs.get_shape()

