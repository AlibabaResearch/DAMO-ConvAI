
from transformers import get_linear_schedule_with_warmup
from lr_scheduler.inverse_square_schedule import get_t5_inverse_square_schedule_with_warmup


def init_lr_scheduler(lr_scheduler_type, optimizer, warmup_steps, max_steps):
    if lr_scheduler_type == 'none':
        return None

    elif lr_scheduler_type == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=warmup_steps,
                                                       num_training_steps=max_steps)

    elif lr_scheduler_type == 'inverse_square':
        lr_scheduler = get_t5_inverse_square_schedule_with_warmup(optimizer=optimizer,
                                                                  num_warmup_steps=warmup_steps)
    else:
        raise NotImplemented

    return lr_scheduler