"""
Running scripts.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from galaxy.args import parse_args
from galaxy.args import str2bool
from galaxy.data.data_loader import DataLoader
from galaxy.data.dataset import Dataset
from galaxy.data.dataset import LazyDataset
from galaxy.data.pretrain_field import PretrainBPETextField
from galaxy.trainers.pretrain_trainer import PretrainTrainer
from galaxy.models.model_base import ModelBase
from galaxy.models.generator import Generator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", type=str2bool, default=False,
                        help="Whether to run trainning.")
    parser.add_argument("--do_test", type=str2bool, default=False,
                        help="Whether to run evaluation on the test dataset.")
    parser.add_argument("--do_infer", type=str2bool, default=False,
                        help="Whether to run inference on the test dataset.")
    parser.add_argument("--num_infer_batches", type=int, default=None,
                        help="The number of batches need to infer.\n"
                        "Stay 'None': infer on entrie test dataset.")
    parser.add_argument("--hparams_file", type=str, default=None,
                        help="Loading hparams setting from file(.json format).")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world_size")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--global_rank", type=int, default=0, help="For distributed training: global_rank")
    PretrainBPETextField.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)
    PretrainTrainer.add_cmdline_argument(parser)
    ModelBase.add_cmdline_argument(parser)
    Generator.add_cmdline_argument(parser)

    hparams = parse_args(parser)
    hparams.use_gpu = False

    torch.cuda.set_device(hparams.local_rank)
    device = torch.device("cuda", hparams.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    hparams.n_gpu = 1
    hparams.world_size = torch.distributed.get_world_size()
    hparams.global_rank = torch.distributed.get_rank()
    print(f"Distributed info: world_size and global rank: " + str(hparams.world_size) +
          "  " + str(hparams.global_rank))
    print(json.dumps(hparams, indent=2))

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)
    hparams.save(os.path.join(hparams.save_dir, "hparams.json"))

    hparams.device = device

    def to_tensor(array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array, device=hparams.device)
        return array

    def set_seed(seed):
        """ fix random seed """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if hparams.n_gpu > 0:
            torch.cuda.manual_seed_all(hparams.seed)

    # set seed
    set_seed(seed=hparams.seed)

    # set reader
    bpe = PretrainBPETextField(hparams)
    hparams.Model.num_token_embeddings = bpe.vocab_size
    hparams.Model.num_turn_embeddings = bpe.max_ctx_turn + 1

    # set generator
    generator = Generator.create(hparams, reader=bpe)

    # loading datasets
    if hparams.do_train:
        train_file = os.path.join(hparams.data_dir, f"{hparams.data_name}.{hparams.tokenizer_type}.jsonl")
        assert os.path.exists(train_file), f"{train_file} isn't exist"
        if hparams.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        train_dataset = LazyDataset(train_file, hparams)
        if hparams.local_rank == 0:
            torch.distributed.barrier()
        train_loader = DataLoader(train_dataset, hparams, collate_fn=bpe.collate_fn_multi_turn)
    else:
        train_loader = []

    # construct Model
    model = ModelBase.create(hparams, generator=generator)
    model.to(hparams.device)

    # multi-gpu setting
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[hparams.local_rank], output_device=hparams.local_rank, find_unused_parameters=True
    )

    # construct Trainer
    trainer = PretrainTrainer(model, to_tensor, hparams, reader=bpe)

    # set optimizer and lr_scheduler
    if hparams.do_train:
        num_batches = len(train_loader)
        trainer.set_optimizers(num_training_steps_per_epoch=num_batches)

    # load model, optimizer and lr_scheduler
    if hparams.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    trainer.load()
    if hparams.local_rank == 0:
        torch.distributed.barrier()

    if hparams.do_train:
        # training process
        init_epoch = trainer.epoch
        for epoch in range(hparams.num_epochs - init_epoch):
            trainer.train_epoch(train_iter=train_loader)


if __name__ == "__main__":
    main()
