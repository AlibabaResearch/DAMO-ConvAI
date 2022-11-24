import copy
import importlib
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import transformers

class Memory(object):
    def __init__(self,buffer=None):
        if buffer is None:
            self.memory = {}
            print('memory model: current memory has saved %d tasks' %
                  len(self.memory.keys()), flush=True)
            total_keys = len(self.memory.keys())
            # print('mem,keys:',self.all_keys,flush=True)
        else:
            self.memory = buffer.memory
            total_keys = len(self.memory.keys())

    def push(self, task_name, value):
        '''
        Add the key-value pairs to the memory dictionary.
        key: task_id
        value: (prompt, learned latent code)
        '''
        self.memory[task_name] = value
        # return
    
    def memory_update(self,):
        return 
    