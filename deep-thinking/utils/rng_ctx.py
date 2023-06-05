import random
import sys

import numpy as np
import torch


class RandomState:
    def __init__(self):
        self.random_mod_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_cpu_state = torch.get_rng_state()
        self.torch_gpu_states = [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]

    def restore(self):
        random.setstate(self.random_mod_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_cpu_state)
        for d, state in enumerate(self.torch_gpu_states):
            torch.cuda.set_rng_state(state, d)


class RandomContext:
    """Save and restore state of PyTorch, NumPy, Python RNGs."""

    def __init__(self, seed=None):
        outside_state = RandomState()

        random.seed(seed)
        np.random.seed(seed)
        if seed is None:
            torch.manual_seed(random.randint(-sys.maxsize - 1, sys.maxsize))
        else:
            torch.manual_seed(seed)
        # torch.cuda.manual_seed_all is called by torch.manual_seed
        self.inside_state = RandomState()

        outside_state.restore()

        self._active = False

    def __enter__(self):
        if self._active:
            raise Exception("RandomContext can be active only once")

        self.outside_state = RandomState()
        self.inside_state.restore()
        self._active = True

    def __exit__(self, exception_type, exception_value, traceback):
        self.inside_state = RandomState()
        self.outside_state.restore()
        self.outside_state = None

        self._active = False


class EmptyContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
