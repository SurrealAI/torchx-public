from test.utils import *
from .test_model_sequential import check_inferred_shape

# print all layer names
# pp(list(Layer.get_registry().keys()))


def test_sequential_spec():
    x = new_tensor((6, 13, 68, 78))

    spec = {
        'type': 'sequential',  # case insensitive
        None: [
            {
                'type': 'Conv2d',
                None: [21, (5, 3)],  # *args
                'padding': (14, 16)
            },
            {
                'type': 'PReLU',
                'shared': False
            },
            {
                'type': 'leakyrelu',
                None: 0.3
            },
            {
                'type': 'maxpool2D',
                **dict(kernel_size=13,
                       stride=(1, 3),
                       dilation=2,
                       padding=(5, 6)),
            }
        ]
    }

    model = Layer.from_spec(spec)

    check_inferred_shape(model, x, 'conv')

    model.add(
        Conv2d(5,
               kernel_size=(5, 11),
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
    shape_gold = check_inferred_shape(model, x, 'conv after add')

    model = Layer.from_spec(model.to_spec())
    shape_roundtrip = check_inferred_shape(model, x, 'conv roundtrip')
    assert U.shape_equals(shape_gold, shape_roundtrip)


def test_rnn_spec():
    x = new_tensor((10, 13, 17))  # batch_size, seq_len, feature_dim

    spec = {
        'type': 'Sequential',
        'layers': [
            {
                'type': 'simplernn',
                None: 23,
                'return_sequences': True,
                'return_state': True,
            },
            {
                'type': 'GetRNNState',
                'mode': 'concat'
            },
            {
                'type': 'ReLU'
            },
        ]
    }

    model = Layer.from_spec(spec)
    shape_gold = check_inferred_shape(model, x, 'RNN')
    model = Layer.from_spec(model.to_spec())
    shape_roundtrip = check_inferred_shape(model, x, 'RNN roundtrip spec')
    assert U.shape_equals(shape_gold, shape_roundtrip)

    nested_spec = {
        'type': 'sequential',
        'layers': [
            {
                'type': 'elu'
            },
            spec,  # from above
            {
                'type': 'gru',
                None: [23],
                **dict(
                    return_sequences=False,
                    return_state=True,
                    num_layers=4,
                    bidirectional=True
                )
            },
            {
                'type': 'GetRNNState',
                'mode': 'h'
            }
        ]
    }
    model = Layer.from_spec(nested_spec)
    check_inferred_shape(model, x, 'RNN (nested sequential)')
    print(model.layer_list)
    print(model)


def test_time_distributed_spec():
    input_shape = (10, 8, 3, 64, 32)  # batch_size, seq_len, image CxHxW
    x = new_tensor(input_shape)

    spec = {
        'type': 'TimeDistributed',
        'layers': [
            dict(
                type='Conv2d',
                out_channels=4,
                kernel_size=(5, 3),
                dilation=2,
                padding=(14, 16)
            ),
            dict(type='PReLU', shared=False),
            dict(
                type='MaxPool2d',
                kernel_size=(5, 3),
                stride=(3, 2),
            ),
            {
                'type': 'LeakyReLU',
                None: 0.3
            }
        ]
    }
    model = Layer.from_spec(spec)
    check_inferred_shape(model, x, 'TimeDistributed')

    model.add(Sequential(
        Flatten(),
        nn.ELU(),
        Dense(47)
    ))
    check_inferred_shape(model, x, 'TimeDistributed flattened')

    # ---------------- Combine with RNN -----------------
    nested_spec = {
        'type': 'Sequential',
        'layer_list': [
            spec,  # TimeDistributed
            dict(
                type='Flatten',
                start_dim=2  # collapse dim[2:]
            ),
            {'type': 'Dense', None: 47},
            dict(
                type='GRU',
                hidden_size=23,
                return_sequences=True,
                return_state=True
            ),
            {'type': 'GetRNNOutput'},
            dict(
                type='LSTM',
                hidden_size=32,
                num_layers=3,
                bidirectional=True,
                return_sequences=True,
                return_state=True
            )
        ]
    }

    model = Layer.from_spec(nested_spec)
    shape_gold = check_inferred_shape(model, x, 'TimeDistributed nested sequential')
    model = Layer.from_spec(model.to_spec())
    shape_roundtrip = check_inferred_shape(model, x, 'TimeDistributed nested roundtrip')
    assert U.shape_equals(shape_gold, shape_roundtrip)
    print(model)
    Layer.set_print_mode('native')
    print(model)
    Layer.set_print_mode('torchx')
