from test.utils import *


def check_inferred_shape(msg, local_dict):
    model = local_dict['model']
    x = local_dict['xs']
    input_shape = local_dict['input_shape']
    inferred_shape = model.get_output_shape(input_shape)

    x = model(x)  # actual forward prop
    actual_shape = U.get_shape(x)
    print(msg, inferred_shape)
    assert U.shape_equals(inferred_shape, actual_shape), \
        ('inferred', inferred_shape, 'actual', actual_shape)


def test_merge_layers():
    shape = (12, 37, 9)
    x = new_variable(shape, 2)
    y = new_variable(shape, 8)
    z = new_variable(shape, 11)

    for MergeCls in [Add, Multiply, Average, Maximum]:
        input_shape = [shape] * 3
        model = MergeCls(input_shape=input_shape)
        xs = x, y, z
        check_inferred_shape(MergeCls.__name__, locals())

    # ---------------- Subtract -----------------
    input_shape = [shape] * 2
    model = Subtract(input_shape=input_shape)
    xs = x, y
    check_inferred_shape('Subtract', locals())

    # ---------------- Concat -----------------
    for axis in [0, 1, 2, -2, -1]:
        input_shape = [shape] * 3
        model = Concat(input_shape=input_shape, axis=axis)
        xs = x, y, z
        out = model(x, y, z)  # also test *varargs
        print('Concat axis=', axis, ':', out.size(), out.mean())
        check_inferred_shape('Concat', locals())


def test_functional():
    x_shape = (12, 31)
    y_shape = (12, 41)

    x = Placeholder(x_shape)
    y = Placeholder(y_shape)
    w = concat(Dense(22)(x), Dense(44)(y))
    w = concat(Dense(33)(x), Dense(55)(w))
    w = concat(Dense(11)(x), Dense(66)(w))
    w = concat(Dense(7)(y), Dense(17)(w))
    out = multiply(subtract(w, Dense(7+17)(w)), Dense(7+17)(concat(x, y)))

    myfunc = Functional(inputs=[x, y], outputs=[out])

    xv = new_variable(x_shape)
    yv = new_variable(y_shape)

    print(myfunc._postorder_traverse())
    myfunc.compile()
    outv = myfunc([xv, yv])
    print(outv.size())

    # myfunc = MyFunc()
    # out = myfunc(xp, yp)
    # print(out.shape)
    # outv = myfunc(xv, yv)
    # print(out.size(), myfunc.get_output_shape([x_shape, y_shape]))


run_all_tests(globals())