import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_t5_inverse_square_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step <= num_warmup_steps:
            return 1.0

        return max(
            0.0, 1.0 / float(current_step)**0.5

        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
