from test.utils import *
from torchx.device import *


def test_devices_to_ids():
    assert ids_to_devices(None) == [torch.device('cpu')]
    assert ids_to_devices('cuda:4') == [torch.device('cuda:4')]
    if not has_cuda():
        assert ids_to_devices('cuda:all') == [torch.device('cpu')]
    assert ids_to_devices(7) == [torch.device('cuda:7')]
    assert ids_to_devices(-1) == [torch.device('cpu')]
    assert ids_to_devices([0, 'cuda:5', torch.device('cpu')]) == \
           [torch.device('cuda:0'), torch.device('cuda:5'), torch.device('cpu')]


def _check_scope(correct_device, correct_dtype):
    x = torch.randn((2, 3))
    assert x.device == correct_device
    assert x.dtype == correct_dtype


def test_default_device_dtype():
    "only works on 4-GPU system"
    if cuda_count() >= 3:
        for device, dtype in [
            (1, torch.float64),
            ('gpu:0', torch.float32),
            (-1, torch.float64),  # CPU
            (2, torch.float32),
            ('cpu', torch.float32),
            ('cuda:1', torch.float64),
        ]:
            set_default_device_dtype(device, dtype)
            device = id_to_device(device)
            _check_scope(device, dtype)
            assert get_default_device_dtype() == (device, dtype)
    else:
        print('TEST ONLY RUNS ON >= 3 GPUs')


def test_device_scope():
    D = torch.device
    with device_scope(-1, torch.double):
        assert get_device_in_scope() == [CPU_DEVICE]
        _check_scope(CPU_DEVICE, torch.double)
    with device_scope(None):
        with device_scope('cpu', torch.double):
            _check_scope(CPU_DEVICE, torch.double)
        assert get_device_in_scope() == [CPU_DEVICE]
        _check_scope(CPU_DEVICE, torch.float32)

    if cuda_count() >= 4:
        with device_scope('cuda:0', torch.double):
            assert get_device_in_scope() == [D('cuda:0')]
            _check_scope(D('cuda:0'), torch.double)

        with device_scope([3, 1, 0, 2]):
            assert get_device_in_scope() == [D('cuda:3'), D('cuda:1'), D('cuda:0'), D('cuda:2')]
            _check_scope(D('cuda:3'), torch.float32)

        with device_scope('cuda:all'):
            _check_scope(D('cuda:0'), torch.float32)

        with device_scope([3, 1], torch.float64):
            with device_scope(2):
                assert get_device_in_scope() == [D('cuda:2')]
                _check_scope(D('cuda:2'), torch.float32)
            _check_scope(D('cuda:3'), torch.float64)

        with device_scope([3, 1], torch.float64):
            with device_scope(2, override_parent=False):
                assert get_device_in_scope() == [D('cuda:3'), D('cuda:1')]
                _check_scope(D('cuda:3'), torch.float64)
            _check_scope(D('cuda:3'), torch.float64)

        with device_scope([3, 1]):
            with device_scope(None, torch.float64):
                assert get_device_in_scope() == [D('cpu')]
                _check_scope(D('cpu'), torch.float64)
            _check_scope(D('cuda:3'), torch.float32)
    else:
        print('TEST ONLY RUNS ON >= 4 GPUs')
