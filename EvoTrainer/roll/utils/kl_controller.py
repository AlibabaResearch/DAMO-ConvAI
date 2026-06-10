import numpy as np
"""
ref: https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/kl_controller.py
"""

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def get_kl_controller(init_kl_coef, target_kl=None, kl_horizon=None):
    if target_kl is None:
        kl_ctrl = FixedKLController(kl_coef=init_kl_coef)
    else:
        kl_ctrl = AdaptiveKLController(init_kl_coef=init_kl_coef, target=target_kl, horizon=kl_horizon)

    return kl_ctrl
