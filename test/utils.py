import os, sys, json, time, re, random, inspect, pickle
from collections import *
from time import sleep
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pprint import pprint

import torchx as tx
import torchx.utils as U


def run_all_tests(globals_dict):
    for k, v in globals_dict.items():
        if (callable(v) and hasattr(v, '__name__')
                and v.__name__.startswith('test_')):
            print('='*7, k, '='*7)
            v()