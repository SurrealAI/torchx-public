"""
torchx.nn.Module and torchx.nn.DataParallel
"""
from test.utils import *
import multiprocessing as mp
import torchx as tx
import torchx.nn as nnx


class MyNet(nnx.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc2 = nn.DataParallel(self.fc2)
        self.fc2 = nnx.DataParallel(self.fc2)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# @pytest.fixture(params=[
#     [3, 2],
#     ['gpu:2'],
#     ['cuda:1', 0, 'gpu:3'],
#     -1,
#     'cuda:all',
# ])
# def devices(request):
#     return request.param

TEST_DEVICES = [
    [3, 2],
    ['gpu:2'],
    ['cuda:3', 0, 'gpu:1'],
    -1,
    'cuda:all',
]


# @pytest.mark.parameterize('dtype', [torch.float, torch.double])
def mytest_data_parallel(devices):
    """
    Make a new process because otherwise cannot empty nvidia-smi
    pytest doesn't work with multiprocessing
    """
    dtype = torch.double
    with tx.device_scope(devices, dtype=dtype):
        net = MyNet(19, 78, 25)
        x = torch.empty(64, 19).uniform_(0, 0.1)
        y = torch.randn(64, 25)
        z = y - net(x)

        devices = tx.ids_to_devices(devices)
        assert z.device == devices[0]
        assert z.dtype == dtype
        # all GPUs without DataParallel allocated should report 0 memory usage
        # os.system('nvidia-smi')
        should_have_mem = [tx.device_to_int(d) for d in devices]
        actual_mems = tx.cuda_memory('all', mode='cache', unit='kb')
        print('IDs', should_have_mem, '\tmemory:', actual_mems)
        for i, actual_mem in enumerate(actual_mems):
            if i in should_have_mem:
                assert actual_mem > 0, ('device', i, actual_mem)
            else:
                assert actual_mem == 0, ('device', i, actual_mem)


for d in TEST_DEVICES:
    p = mp.Process(target=mytest_data_parallel, args=(d,))
    p.start()
    p.join()
