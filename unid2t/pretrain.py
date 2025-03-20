import sys
import os
import torch
import numpy as np

from opts import init_opts_for_pretraining
from trainer import Trainer


def main():
    args = init_opts_for_pretraining()
    trainer = Trainer.init_trainer(args)

    trainer.train()


if __name__ == '__main__':
    main()

