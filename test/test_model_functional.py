from test.utils import *


def check_inferred_shape(model, xs, msg):
    input_shape = U.get_shape(xs)
    z = model(xs)  # actual forward prop
    actual_shape = U.get_shape(z)
    inferred_shape = model.get_output_shape(input_shape)
    print(msg, inferred_shape)

    # backward on all output tensors
    if isinstance(z, (list, tuple)):
        # output list, backward on every list item
        for z_, shape_ in zip(z, actual_shape):
            z_.backward(torch.randn(shape_), retain_graph=True)
    elif isinstance(z, dict):
        # output dict, backward on every dict entry
        for key in z:
            z_, shape_ = z[key], actual_shape[key]
            z_.backward(torch.randn(shape_), retain_graph=True)
    else:
        z.backward(torch.randn(actual_shape))

    # check all input tensors should have gradient
    if isinstance(xs, (list, tuple)):
        pass
    elif isinstance(xs, dict):
        xs = xs.values()
    else:
        xs = [xs]

    for x in xs:
        assert x.grad is not None, 'backprop does not reach the input tensor'
        x.grad = None  # clear grad for the next round of testing
    assert U.shape_equals(inferred_shape, actual_shape), \
        ('inferred', inferred_shape, 'actual', actual_shape)


def test_merge_layers():
    shape = (12, 37, 9)
    x = fill_tensor(shape, 2)
    y = fill_tensor(shape, 8)
    z = fill_tensor(shape, 11)

    for MergeCls in [Add, Multiply, Average, Maximum]:
        check_inferred_shape(
            MergeCls(), [x, y, z], MergeCls.__name__
        )

    # ---------------- Subtract -----------------
    check_inferred_shape(
        Subtract(), [x, y], 'Subtract'
    )

    # ---------------- Concat -----------------
    for axis in [0, 1, 2, -2, -1]:
        concat = Concat(axis=axis)
        xs = x, y, z
        out = concat(*xs)  # also test *varargs
        assert nnx.th_to_scalar(out.mean()) == 7.
        check_inferred_shape(
            concat, [x, y, z], 'Concat'
        )


def test_placeholder_overload():
    xshape = (5, 6)
    yshape = (7, 8)
    x = Placeholder(xshape)
    y = Placeholder(yshape)
    # tries indexing using integer and slice
    out = x + y[2:, 2:] - x * y[:-2, :-2]
    model = Functional(inputs=[x, y], outputs=[out])

    xv = fill_tensor(xshape, 3)
    yv = fill_tensor(yshape, 5)

    # should be add, multiply, subtract
    print(model.postorder_traverse())
    outv = model(xv, yv)
    assert isinstance(outv, list)
    assert torch.equal(outv[0], fill_tensor(xshape, -7))
    check_inferred_shape(model, [xv, yv], 'placeholder overload')


def test_placeholder_reshape():
    x = Placeholder((5, 2, 3, 4))
    y = Placeholder((5, 2, 3*4))

    for out in [
        x.flatten(start_dim=2) + y,
        x * y.view(5, 2, -1, 4),
        x * y.view(-1, 2, 3, 4),
        x.flatten() + y.flatten(),
        x.flatten().view(-1, 4) - y.flatten(start_dim=0).view(5*2*3, -1)
    ]:
        model = Functional(inputs=[x, y], outputs=out)

        xv = fill_tensor(x.shape, 3)
        yv = fill_tensor(y.shape, 5)
        check_inferred_shape(model, [xv, yv], 'placeholder reshape')


def single_node_testcases(input_case_id, output_case_id):
    """
    provide cartesian-product test cases for test_single_node
    """
    x_shape = (12, 31)
    y_shape = (12, 41)

    x = Placeholder(x_shape)
    y = Placeholder(y_shape)
    w1 = concat(Dense(22)(x), Sigmoid()(Dense(44)(y)))
    w1 = PReLU(shared=False)(w1)
    assert w1.shape == (12, 66)
    w2 = concat(Dense(33)(w1), Dense(55)(w1[:, :33] / w1[:, 33:]))
    assert w2.shape == (12, 88)
    w3 = concat(Dense(11)(w2), Dense(66)(w2[:, 10:40] / w2[:, 20:50]))
    assert w3.shape == (12, 77)
    w4 = (concat(Dense(7)(w3 * w3), Dense(17)(w3)))
    w4 = RReLU(1/15, 1/7)(w4)
    assert w4.shape == (12, 24)
    out1 = multiply(subtract(w4, Dense(7+17)(w3)), Dense(7+17)(concat(w4, w3, w4)))
    assert out1.shape == (12, 24)
    out2 = maximum(w4, w4 * w4, Dense(7+17)(w3), Dense(7+17)(concat(w3, w4)))
    assert out2.shape == (12, 24)

    input_test_cases = [
        [x, y],
        w1,
        [w2],
        {'myx': x, 'myy': y},
        {'myw3': w3}
    ]
    output_test_cases = [
        out1,
        [out2],
        [w3, w3 + w3, w3 * w3, w3, minimum(w3, w3)],
        {'myout1': out1},
        {'myout2': out2, 'myout1': out1, 'myout2_again': out2},
    ]
    return input_test_cases[input_case_id], output_test_cases[output_case_id]


@pytest.mark.parametrize('output_case_id', range(5))
@pytest.mark.parametrize('input_case_id', range(5))
def test_single_node(input_case_id, output_case_id):
    "single node computation graph"
    inputs, outputs = single_node_testcases(input_case_id, output_case_id)
    model = Functional(inputs=inputs, outputs=outputs)
    input_tensors = randn_pstruct(inputs)
    check_inferred_shape(model, input_tensors,
                         'inputs={} outputs={}'.format(inputs, outputs))


def multi_node_testcases(input_case_id, output_case_id):
    """
    provide cartesian-product test cases for test_multi_node

    Set up a complicated computation graph with shared layers
    """
    x_shape = (12, 31)
    y_shape = (12, 41)

    x0 = Placeholder(x_shape)
    y0 = Placeholder(y_shape)

    x1 = Dense(22)(x0)
    y1 = Dense(22)(y0)
    shared1 = Dense(17)
    out1 = ReLU6()(concat(shared1(x1), shared1(y1), shared1(x1)))
    assert out1.shape == (12, 51)
    concat_xyx = concat(x1, y1, x1)[:, :17*3]
    assert concat_xyx.shape == (12, 51)
    concat_yxy = concat(y1, x1, y1)[:, -17*3:]
    concat_yxy = PReLU(shared=False)(concat_yxy)
    assert concat_yxy.shape == (12, 51)
    shared2 = Dense(47)
    shared3 = Dense(47)
    out2 = average(
        PReLU()(shared2(concat_xyx)),
        shared2(out1),
        shared2(concat_xyx),
        SELU()(shared3(out1 + concat_yxy + out1)),
        shared2(concat_yxy / out1),
        shared2(out1 - concat_xyx),
        shared3(concat_yxy * out1 * concat_yxy),
        shared3(out1),
    )
    assert out2.shape == (12, 47)

    input_test_cases = [
        [x0, y0],
        [x1, y0],
        [x0, y1],
        [y1, x1],
        {'myx': x0, 'myy': y0},
        {'myx': x1, 'myy': y1},
    ]
    output_test_cases = [
        out2,
        [out2, out1],
        [out1, out1, out2, out2],
        {'myout1': out1},
        {'myout2': out2, 'myout1': out1, 'myout2_again': out2},
        {'yxy': concat_yxy, 'xyx': concat_xyx},
    ]
    return input_test_cases[input_case_id], output_test_cases[output_case_id]


@pytest.mark.parametrize('output_case_id', range(6))
@pytest.mark.parametrize('input_case_id', range(6))
def test_multi_node(input_case_id, output_case_id):
    inputs, outputs = multi_node_testcases(input_case_id, output_case_id)
    model = Functional(inputs=inputs, outputs=outputs)
    input_tensors = randn_pstruct(inputs)
    check_inferred_shape(model, input_tensors,
                         'inputs={} outputs={}'.format(inputs, outputs))

# run_all_tests(globals())
