from test.utils import *

# pprint(get_layer_registry())


def check_inferred_shape(model, x, msg):
    assert x.grad is None
    input_shape = U.get_shape(x)
    inferred_shape = model.get_output_shape(input_shape)

    z = model(x)  # actual forward prop
    actual_shape = U.get_shape(z)
    if isinstance(z, (list, tuple)):
        z[0].backward(torch.randn(actual_shape[0]))
    else:
        z.backward(torch.randn(actual_shape))
    assert x.grad is not None, 'backprop does not reach the input tensor'
    x.grad = None
    print(msg, inferred_shape)
    assert U.shape_equals(inferred_shape, actual_shape), \
        ('inferred', inferred_shape, 'actual', actual_shape)


def test_dense_sequential():
    input_shape = (12, 37)
    x = new_tensor(input_shape)

    model = Sequential(
        Dense(93),
        nn.ReLU(),  # builtin pytorch, will be auto-wrapped into torchx.Layer
        Dense(42),
        LeakyReLU(0.1),  # torchx.layers
        Dense(77)
    )
    check_inferred_shape(model, x, 'dense')


def test_conv_sequential():
    input_shape= (12, 13, 128, 256)
    x = new_tensor(input_shape)

    model = Sequential([
        Conv2d(21,
               kernel_size=(5, 3),
               padding=(14, 16)),
        PReLU(shared=False),
        Conv2d(15,
               kernel_size=(3, 17),
               stride=(4, 5),
               padding=35),
        nn.LeakyReLU(0.3),
        MaxPool2d(kernel_size=13,
                  stride=(1, 3),
                  dilation=2,
                  padding=(5, 6)),
    ])
    check_inferred_shape(model, x, 'conv before flatten')

    model.add(
        Conv2d(5,
               kernel_size=(15, 11),
               stride=2,
               dilation=(3, 4),
               padding=45),
        LeakyReLU(0.1),
        AvgPool2d((2, 3),
                  stride=None,
                  padding=1),
        ELU(),
        Flatten(),
    )
    check_inferred_shape(model, x, 'conv after flatten')

    model.add(Sequential([
        Dense(9),
        Flatten(),  # should have no effect
        nn.ELU(),
    ]))
    model.add(Dense(57))

    check_inferred_shape(model, x, 'after dense')


def test_rnn_without_state():
    input_shape = (10, 8, 17)  # batch_size, seq_len, feature_dim
    x = new_tensor(input_shape)

    model = Sequential([
        SimpleRNN(23,
                  return_sequences=True),
        SELU(),
        GetRNNOutput(),  # essentially no-op
    ])
    check_inferred_shape(model, x, 'SimpleRNN, seq=True')

    model = Sequential(
        GRU(23,
            return_sequences=False,
            num_layers=2,
            bidirectional=True),
        Tanh(),
    )
    check_inferred_shape(model, x, 'GRU, seq=False')

    model = Sequential([
        LSTM(23,
             return_sequences=True,
             num_layers=3,
             bidirectional=True),
        nn.ReLU(),
    ])
    check_inferred_shape(model, x, 'LSTM, seq=True')


def test_rnn_with_state():
    input_shape = (10, 13, 17)  # batch_size, seq_len, feature_dim
    x = new_tensor(input_shape)

    model = Sequential([
        SimpleRNN(23,
                  return_sequences=True,
                  return_state=True),
    ])
    check_inferred_shape(model, x, 'SimpleRNN, seq=True')
    model.add(GetRNNState(mode='concat'))
    model.add(nn.ReLU())
    check_inferred_shape(model, x, 'SimpleRNN, mode=concat')

    model = Sequential(
        GRU(23,
            return_sequences=False,
            return_state=True,
            num_layers=4,
            bidirectional=True),
    )
    check_inferred_shape(model, x, 'GRU, seq=False')
    model.add([
        GetRNNState(mode='h'),
        nn.ReLU()
    ])
    check_inferred_shape(model, x, 'GRU, mode=h')

    model = Sequential(
        GRU(23,
            return_sequences=True,
            return_state=True,
            num_layers=4,
            bidirectional=True),
        GetRNNOutput(),
        PReLU(shared=False)
    )
    check_inferred_shape(model, x, 'GRU, output')

    model = Sequential([
        LSTM(23,
             return_sequences=True,
             return_state=True,
             num_layers=3,
             bidirectional=True),
    ])
    check_inferred_shape(model, x, 'LSTM, seq=True')
    model.add([
        GetRNNState(mode='c'),
        PReLU()
    ])
    check_inferred_shape(model, x, 'LSTM, mode=c')

    model = Sequential([
        LSTM(23,
             return_sequences=False,
             return_state=True,
             num_layers=1,
             bidirectional=True),
    ])
    check_inferred_shape(model, x, 'LSTM, seq=False')
    model.add([
        GetRNNState(mode='concat'),
        nn.ReLU()
    ])
    check_inferred_shape(model, x, 'LSTM, mode=concat')


def test_time_distributed():
    input_shape = (10, 8, 3, 64, 32)  # batch_size, seq_len, image CxHxW
    x = new_tensor(input_shape)

    model = TimeDistributed(
        Conv2d(4,
               kernel_size=(5, 3),
               dilation=2,
               padding=(14, 16)),
        PReLU(shared=False),
        Conv2d(2,
               kernel_size=(1, 3),
               stride=(2, 1),
               padding=10),
        nn.LeakyReLU(0.3),
    )

    check_inferred_shape(model, x, 'TimeDistributed conv')

    model.add(
        MaxPool2d(kernel_size=(5, 3),
                  stride=(3, 2)),
    )
    check_inferred_shape(model, x, 'TimeDistributed conv add again')

    model.add(Sequential(
        Flatten(),
        nn.ELU(),
        Dense(47)
    ))
    check_inferred_shape(model, x, 'TimeDistributed flattened')

    # ---------------- Combine with RNN -----------------
    time_distributed_layer = model
    model = Sequential([
        time_distributed_layer,
        GRU(23,
            return_sequences=True,
            return_state=True),
        GetRNNOutput(),
        nn.ELU(),
        LSTM(33,
             num_layers=3,
             bidirectional=True,
             return_sequences=True,
             return_state=True)
    ])
    check_inferred_shape(model, x, 'TimeDistributed+LSTM')

    model.add(
        GetRNNOutput(),
        TimeDistributed([
            Dense(11),
            nn.LeakyReLU(0.1),
            Dense(51),
        ]),
        nn.ReLU(),
    )
    check_inferred_shape(model, x, 'TimeDistributed again')

    model.add(Sequential(
        TimeDistributed([
            Dense(61),
            nn.Sigmoid(),
            Dense(31),
        ]),
        LSTM(23,
             num_layers=6,
             return_state=True,
             bidirectional=True),
        GetRNNState(mode='concat'),
        Flatten(),
    ))
    check_inferred_shape(model, x, 'TimeDistributed final state concat')


# run_all_tests(globals())
