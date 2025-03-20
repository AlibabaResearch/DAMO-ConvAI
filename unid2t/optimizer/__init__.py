import torch.optim as torch_optim

from transformers import AdamW, Adafactor
from fairseq.optim.adafactor import Adafactor as Fairseq_Adafactor


def init_optimizer(model, learner, learning_rate):
    # learner = args.learner.lower()
    # learning_rate = float(args.learning_rate)
    if learner == 'adamw':
        optimizer = AdamW(params=model.parameters(), lr=learning_rate)
    elif learner == 'adam':
        optimizer = torch_optim.Adam(params=model.parameters(), lr=learning_rate)
    elif learner == 'adagrad':
        optimizer = torch_optim.Adagrad(params=model.parameters(), lr=learning_rate)
    elif learner == 'adafactor':
        optimizer = Adafactor(params=model.parameters(), lr=learning_rate,
                              scale_parameter=False, relative_step=False)
    elif learner == 'fairseq_adafactor':
        optimizer = Fairseq_Adafactor(params=model.parameters(), lr=learning_rate,
                                      scale_parameter=False, relative_step=False)
    else:
        raise NotImplemented

    return optimizer
