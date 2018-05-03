from test.utils import *

# pprint(get_layer_registry())


def check_inferred_shape(msg, local_dict):
    model = local_dict['model']
    x = local_dict['x']
    assert x.grad is None
    input_shape = local_dict['input_shape']
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

    model = Sequential([
        Dense(93),
        nn.ReLU(),
        Dense(42),
        nn.LeakyReLU(0.1),
        Dense(77)
    ])
    check_inferred_shape('dense', locals())


def test_conv_sequential():
    input_shape= (12, 13, 128, 256)
    x = new_tensor(input_shape)

    model = Sequential([
        Conv2d(21,
               kernel_size=(5, 3),
               padding=(14, 16)),
        nn.ReLU(),
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
    check_inferred_shape('conv before flatten', locals())

    model.add([
        Conv2d(5,
               kernel_size=(15, 11),
               stride=2,
               dilation=(3, 4),
               padding=45),
        nn.LeakyReLU(0.1),
        AvgPool2d((2, 3),
                  stride=None,
                  padding=1),
        nn.ELU(),
        Flatten(),
    ])
    check_inferred_shape('conv after flatten', locals())

    model.add(Sequential([
        Dense(9),
        Flatten(),  # should have no effect
        nn.ELU(),
    ]))
    model.add(Dense(57))

    check_inferred_shape('after dense', locals())


def test_rnn_without_state():
    input_shape = (10, 8, 17)  # batch_size, seq_len, feature_dim
    x = new_tensor(input_shape)

    model = Sequential([
        SimpleRNN(23,
                  return_sequences=True),
        nn.ReLU(),
        GetRNNOutput(),  # essentially no-op
    ])
    check_inferred_shape('SimpleRNN, seq=True', locals())

    model = Sequential([
        GRU(23,
            return_sequences=False,
            num_layers=2,
            bidirectional=True),
        nn.ReLU(),
    ])
    check_inferred_shape('GRU, seq=False', locals())

    model = Sequential([
        LSTM(23,
             return_sequences=True,
             num_layers=3,
             bidirectional=True),
        nn.ReLU(),
    ])
    check_inferred_shape('LSTM, seq=True', locals())


def test_rnn_with_state():
    input_shape = (10, 13, 17)  # batch_size, seq_len, feature_dim
    x = new_tensor(input_shape)

    model = Sequential([
        SimpleRNN(23,
                  return_sequences=True,
                  return_state=True),
    ])
    check_inferred_shape('SimpleRNN, seq=True', locals())
    model.add(GetRNNState(mode='concat'))
    model.add(nn.ReLU())
    check_inferred_shape('SimpleRNN, mode=concat', locals())

    model = Sequential([
        GRU(23,
            return_sequences=False,
            return_state=True,
            num_layers=4,
            bidirectional=True),
    ])
    check_inferred_shape('GRU, seq=False', locals())
    model.add([
        GetRNNState(mode='h'),
        nn.ReLU()
    ])
    check_inferred_shape('GRU, mode=h', locals())

    model = Sequential([
        GRU(23,
            return_sequences=True,
            return_state=True,
            num_layers=4,
            bidirectional=True),
        GetRNNOutput(),
        nn.ReLU()
    ])
    check_inferred_shape('GRU, output', locals())

    model = Sequential([
        LSTM(23,
             return_sequences=True,
             return_state=True,
             num_layers=3,
             bidirectional=True),
    ])
    check_inferred_shape('LSTM, seq=True', locals())
    model.add([
        GetRNNState(mode='c'),
        nn.ReLU()
    ])
    check_inferred_shape('LSTM, mode=c', locals())

    model = Sequential([
        LSTM(23,
             return_sequences=False,
             return_state=True,
             num_layers=1,
             bidirectional=True),
    ])
    check_inferred_shape('LSTM, seq=False', locals())
    model.add([
        GetRNNState(mode='concat'),
        nn.ReLU()
    ])
    check_inferred_shape('LSTM, mode=concat', locals())


def test_time_distributed():
    input_shape = (10, 8, 3, 64, 32)  # batch_size, seq_len, image CxHxW
    x = new_tensor(input_shape)

    model = TimeDistributed([
        Conv2d(4,
               kernel_size=(5, 3),
               dilation=2,
               padding=(14, 16)),
        nn.ReLU(),
        Conv2d(2,
               kernel_size=(1, 3),
               stride=(2, 1),
               padding=10),
        nn.LeakyReLU(0.3),
    ])

    check_inferred_shape('TimeDistributed conv', locals())

    model.add(
        MaxPool2d(kernel_size=(5, 3),
                  stride=(3, 2)),
    )
    check_inferred_shape('TimeDistributed conv add again', locals())

    model.add(Sequential([
        Flatten(),
        nn.ELU(),
        Dense(47)
    ]))
    check_inferred_shape('TimeDistributed flattened', locals())

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
    check_inferred_shape('TimeDistributed+LSTM', locals())

    model.add([
        GetRNNOutput(),
        TimeDistributed([
            Dense(11),
            nn.LeakyReLU(0.1),
            Dense(51),
        ]),
        nn.ReLU(),
    ])
    check_inferred_shape('TimeDistributed again', locals())

    model.add(Sequential([
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
    ]))
    check_inferred_shape('TimeDistributed final state concat', locals())


# run_all_tests(globals())
