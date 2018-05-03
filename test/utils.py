import os, sys, json, time, re, random, inspect, pickle
from collections import *
from time import sleep
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torchx as tx
import torchx.utils as U
from torchx.layers import *
import torchx.nn as nnx
import pytest

pp = pprint.pprint


def run_all_tests(globals_dict):
    for k, v in globals_dict.items():
        if (callable(v) and hasattr(v, '__name__')
                and v.__name__.startswith('test_')):
            print('='*7, k, '='*7)
            v()


def new_tensor(*input_shape, requires_grad=True):
    return torch.randn(*input_shape, requires_grad=requires_grad)


def fill_tensor(input_shape, value, requires_grad=True):
    """
    Alternative way:

    x = torch.ones(input_shape) * value
    x = x.detach()  # otherwise will be a multiplication node with no gradient
    x.requires_grad = requires_grad
    """
    return nnx.th_new_full(input_shape, value, requires_grad=requires_grad)


def randn_pstruct(struct):
    """
    Fill a struct of placeholders with randn tensors
    """
    return U.recursive_map(
        struct=struct,
        func=lambda p: new_tensor(p.shape),
    )