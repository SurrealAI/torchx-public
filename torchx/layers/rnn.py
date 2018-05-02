import torch
import torch.nn as nn
from .core import Layer
import torchx.utils as U


# TODO: does not support Keras `stateful` parameter yet
# TODO: h_0 and c_0 defaults to all zeros
# TODO: add batch_first arg back as we support batch dim
class RNNBase(Layer):
    def __init__(self, rnn_class,
                 rnn_state_arity,
                 hidden_size,
                 *, input_shape=None,
                 return_sequences=False,
                 return_state=False,
                 bidirectional=False,
                 num_layers=1,
                 **kwargs):
        """
        Similar to Keras recurrent layer API. `batch_first` is always True.

        Args:
          rnn_class: modules derive from nn.modules.rnn.RNNBase
          rnn_state_arity:
            1 for SimpleRNN and GRU (just "H")
            2 for LSTM ("H" and "C")
          input_shape: [batch_size, sequence_length, feature]
          return_sequences: same as Keras API
          return_state: same as Keras API
              https://keras.io/layers/recurrent/#rnn

        Returns:
        - if return_state: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states, each with shape (batch_size, units).
        - if return_sequences: 3D tensor with shape (batch_size, timesteps, units).
        - else, 2D tensor with shape (batch_size, units).

        Examples:
          For 3-layer bidirectional LSTM with "H" and "C" internal states
          input shape [batch_size, sequence_length, feature] == [10, 17, 200]
          state shape is 3 layers x bidirectional = 6
          hidden_size = 90
          return_state=True AND return_sequences=True,
            list of 3 tensors, [output=(10, 17, 90), H=(10, 6, 90), C=(10, 6, 90)]
          return_state=True AND return_sequences=False
            list of 3 tensors, [output=(10, 90), H=(10, 6, 90), C=(10, 6, 90)]
          return_state=False AND return_sequences=True
            single tensor, (10, 17, 90)
          return_state=False AND return_sequences=False
            single tensor, (10, 90)
        """
        super().__init__(input_shape=input_shape, **kwargs)
        self.RNNClass = rnn_class
        self.rnn_state_arity = rnn_state_arity
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        assert 'batch_first' not in kwargs, 'batch_first is always set to True'
        self.return_sequences = return_sequences
        self.return_state = return_state

    def _build(self, input_shape):
        assert len(input_shape) == 3
        _, seq_len, features = input_shape
        self.rnn = self.RNNClass(
            input_size=features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            **self.init_kwargs
        )

    def forward(self, x):
        output, states = self.rnn(x)
        if not isinstance(states, tuple):
            states = (states,)
        assert len(states) == self.rnn_state_arity, 'wrong rnn_state_arity'
        if not self.return_sequences:
            # return only the last output timestep
            output = output[:, -1, :]
        if self.return_state:
            # PyTorch always puts [num_layer] dim before batch dim for RNN states,
            # even if you specify batch_first=True
            # i.e. [num_layer, batch, hidden_size]
            states = tuple(state.transpose(0, 1) for state in states)
            return (output,) + states
        else:
            return output

    def get_output_shape(self, input_shape):
        """
        Returns:
            tuple (output, H, C) if return_state=True
        """
        assert len(input_shape) == 3
        batch_size, seq_len, _ = input_shape
        num_directions = 2 if self.bidirectional else 1
        out_features = self.hidden_size * num_directions
        state_dim = self.num_layers * num_directions

        if self.return_sequences:
            output_shape = (batch_size, seq_len, out_features)
        else:
            output_shape = (batch_size, out_features, )
        if self.return_state:
            state_shape = (batch_size, state_dim, self.hidden_size)
            # LSTM has two states, "H" and "C"
            state_shapes = [state_shape for _ in range(self.rnn_state_arity)]
            return (output_shape, *state_shapes)
        else:
            return output_shape


class SimpleRNN(RNNBase):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(
            rnn_class=nn.RNN,
            rnn_state_arity=1,
            hidden_size=hidden_size,
            **kwargs
        )


class GRU(RNNBase):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(
            rnn_class=nn.GRU,
            rnn_state_arity=1,
            hidden_size=hidden_size,
            **kwargs
        )


class LSTM(RNNBase):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(
            rnn_class=nn.LSTM,
            rnn_state_arity=2,
            hidden_size=hidden_size,
            **kwargs
        )


class GetRNNOutput(Layer):
    """
    Get an RNNBase layer's output tensor

    Examples:
        If you set return_state=True, LSTM will return (output, state_h, state_c)
        GetRNNOutput ensures that you always receive the output tensor only
    """
    def __init__(self, *, input_shape=None):
        super().__init__(input_shape=input_shape)

    def _build(self, input_shape):
        pass

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            return x[0]
        else:
            return x

    def get_output_shape(self, input_shape):
        if U.is_multi_shape(input_shape):
            # input_shape is a nested tuple, means upstream return_state=True
            return input_shape[0]
        else:
            assert U.is_simple_shape(input_shape)
            return input_shape


class GetRNNState(Layer):
    """
    Get an RNNBase layer's state tensor

    Examples:
        If you set return_state=True, LSTM will return (output, state_h, state_c)
        GetRNNState ensures that you always receive the state tensor only
    """
    MODES = ['h', 'c', 'concat']

    def __init__(self, mode='concat',
                 *, input_shape=None):
        """
        Args:
          mode: state process mode
          - "h": return hidden state tensor
          - "c": return cell state tensor (LSTM only)
          - "concat": concat h and c along the last feature dimension.
            if not LSTM, simply return h
        """
        super().__init__(input_shape=input_shape)
        self.mode = mode.lower()
        assert self.mode in self.MODES, ('valid modes are', self.MODES)

    def _build(self, input_shape):
        pass

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            if self.mode == 'h':
                return x[1]
            elif self.mode == 'c':
                assert len(x) == 3, \
                    'upstream RNN is not LSTM, cannot use "c" mode in GetRNNState'
                return x[2]
            elif self.mode == 'concat':
                if len(x) == 3:  # LSTM concat h and c along the last dim
                    h = x[1]
                    c = x[2]
                    return torch.cat([h, c], dim=-1)
                else:  # not LSTM, simply return h
                    return x[1]
        else:
            raise ValueError('No RNN state passed, '
                         'please set return_state=True in upstream RNN Layer')

    def get_output_shape(self, input_shape):
        if U.is_multi_shape(input_shape):
            # input_shape is a nested tuple, means upstream return_state=True
            if self.mode == 'h':
                return input_shape[1]
            elif self.mode == 'c':
                assert len(input_shape) == 3, \
                    'upstream RNN is not LSTM, cannot use "c" mode in GetRNNState'
                return input_shape[2]
            elif self.mode == 'concat':
                if len(input_shape) == 3:  # LSTM concat h and c along the last dim
                    h_shape = input_shape[1]
                    c_shape = input_shape[2]
                    assert len(h_shape) == len(c_shape), 'internal error'
                    concat_features = h_shape[-1] + c_shape[-1]
                    return h_shape[:-1] + (concat_features,)
                else:  # not LSTM, simply return shape of h
                    return input_shape[1]
        else:
            raise ValueError('No RNN state shape passed, '
                         'please set return_state=True in upstream RNN Layer')

