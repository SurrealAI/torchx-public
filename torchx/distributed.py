"""
https://pytorch.org/docs/stable/distributed.html
https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
https://pytorch.org/docs/stable/distributed.html#launch-utility

Complete working example:
https://github.com/pytorch/examples/blob/master/imagenet/main.py

Wrapper around nn.DataParallel and nn.parallel.DistributedDataParallel
"""
import os
import sys
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.distributed
import contextlib
import torchx.device as _txd


def DataParallel(module, output_device=None, dim=0):
    """
    Reads the device from torchx.device_scope()
    If the device in scope is CPU, this wrapper will be no op.
    """
    devices, dtype = _txd.get_torchx_device_dtype()
    if devices[0] == torch.device('cpu'):
        return module
    else:
        device_ids = [_txd.device_to_int(dev) for dev in devices]
        return nn.DataParallel(
            module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim
        )


def DistributedDataParallel(module, dim=0):
    """
    Launch utility (v0.4)
    https://pytorch.org/docs/stable/distributed.html#launch-utility
    https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

    DistributedDataParallel (GPU)
    https://pytorch.org/docs/stable/nn.html#distributeddataparallel

    DistributedDataParallelCPU (CPU)
    https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed_cpu.py

    Warnings:
        GPU mode only works with `gloo` and `nccl` backend,
        CPU mode only works with `mpi`, `tcp`, and `gloo` backend,
        If you run with `gloo` backend, you'll see this issue
        https://github.com/pytorch/pytorch/issues/2530
        It doesn't affect the result at all but prints error messages.
    """
    devices, dtype = _txd.get_torchx_device_dtype()
    assert len(devices) == 1, \
        'DistributedDataParallel should work with only one device in scope. ' \
        'Please use either torchx.DataParallel or enclose in the context manager ' \
        '`DistributedManager.device_scope()` instead.'
    device = devices[0]
    if device == torch.device('cpu'):
        return nn.parallel.DistributedDataParallelCPU(module)
    else:
        device_id = _txd.device_to_int(device)
        return nn.parallel.DistributedDataParallel(
            module,
            device_ids=[device_id],
            output_device=device_id,
            dim=dim
        )


class DistributedManager:
    LOCAL_RANK_ENV_NAME = 'TORCHX_LOCAL_RANK'

    def __init__(self,
                 backend,
                 num_procs,
                 num_nodes=1,
                 node_rank=0,
                 master_addr='localhost',
                 master_port=23333):
        """
        https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
        Uses the `env://` init method only.

        Run your main script with `-h` to show

        Args:
            backend
            num_procs: number of processes *per node*
            num_nodes:
            node_rank:
            master_addr:
            master_port:
        """
        parser = argparse.ArgumentParser('TorchX distributed manager')
        parser.add_argument(
            "--num-nodes", type=int, default=num_nodes,
            help="The number of nodes to use for distributed training"
        )
        parser.add_argument(
            "--node-rank", type=int, default=node_rank,
            help="The rank of the node for multi-node distributed training"
        )
        parser.add_argument(
            "--num-procs", type=int, default=num_procs,
            help="The number of processes *per node*, "
                 "for GPU training, this is recommended to be set "
                 "to the number of GPUs in your system so that "
                 "each process can be bound to a single GPU."
        )
        if master_addr.lower() == 'localhost':
            master_addr = '127.0.0.1'
        parser.add_argument(
            "--master-addr", type=str, default=master_addr,
            help="Master node (rank 0)'s address, should be either "
                 "the IP address or the hostname of node 0, for "
                 "single node multi-proc training, the "
                 "--master_addr can simply be 127.0.0.1"
        )
        parser.add_argument(
            "--master-port", type=int, default=master_port,
            help="Master node (rank 0)'s free port that needs to "
                 "be used for communciation during distributed training"
        )
        self.parser = parser
        self.backend = backend.lower()
        self._default_dist_config = {
            'num_procs': num_procs,
            'num_nodes': num_nodes,
            'node_rank': node_rank,
            'master_addr': master_addr,
            'master_port': master_port
        }

    def get_parser(self):
        return self.parser

    def _worker_setup(self):
        torch.distributed.init_process_group(self.backend)

    def entry(self):
        """
        """
        # DistributedManager master proc will set an env variable for worker proc
        # if this variable doesn't exist, that means it's the master proc
        # otherwise it's worker process, entry() will be no-op
        TORCHX_FLAG = '_TORCHX_DISTRIBUTED_WORKER_'
        if TORCHX_FLAG in os.environ:
            # entering worker proc
            self._worker_setup()
            return

        # entering master proc
        # master proc will launch worker procs using `subprocess`
        args = self.parser.parse_args()

        # world size in terms of number of processes
        dist_world_size = args.num_procs * args.num_nodes

        # set PyTorch distributed related environmental variables
        current_env = os.environ.copy()
        current_env["MASTER_ADDR"] = args.master_addr
        current_env["MASTER_PORT"] = str(args.master_port)
        current_env["WORLD_SIZE"] = str(dist_world_size)
        current_env[TORCHX_FLAG] = '0'  # dummy value

        procs = []

        for local_rank in range(0, args.num_procs):
            # each process's rank
            dist_rank = args.num_procs * args.node_rank + local_rank
            current_env["RANK"] = str(dist_rank)
            current_env[self.LOCAL_RANK_ENV_NAME] = str(local_rank)

            # sys.argv contains this script's name
            cmd = [sys.executable, '-u', *sys.argv]
            proc = subprocess.Popen(cmd, env=current_env)
            procs.append(proc)

        for proc in procs:
            proc.wait()
        sys.exit(0)  # master proc exit without proceeding

    def local_rank(self):
        return int(os.environ[self.LOCAL_RANK_ENV_NAME])

    def device_scope(self):
        return _txd.device_scope(self.local_rank())
