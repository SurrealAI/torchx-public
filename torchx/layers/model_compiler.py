import torch.nn as nn
import surreal.utils as U



class CompiledModule(U.Module):
    def __init__(self, inputs,
                 body,
                 outputs,
                 input_format,
                 output_format,
                 ):
        pass


class ModelCompiler:
    def __init__(self, config, input_shapes):
        """
        Args:
          config: dict needs to have 4 sections
          - inputs: input heads, process multimodal inputs
          - body: merge processed inputs and pass through a feed forward (Sequential) NN
          - outputs: output heads
          - global: how the input and body connects,
              format of input and output (dict or tuple), etc.
          input_shapes: for shape inference. Must have the same format (dict or tuple)
            as specified in config['global']
        """
        for key in ['inputs', 'body', 'outputs', 'global']:
            assert key in config, key + ' section do not exist'


