"""
Running scripts.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from space.args import parse_args
from space.args import str2bool
from space.data.data_loader import get_sequential_data_loader
from space.data.dataset import Dataset
from space.data.fields.intent_field import IntentBPETextField
from space.models.model_base import ModelBase
from space.models.generator import Generator
from space.trainers.intent_trainer import IntentTrainer


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
    IntentBPETextField.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)
    IntentTrainer.add_cmdline_argument(parser)
    ModelBase.add_cmdline_argument(parser)
    Generator.add_cmdline_argument(parser)

    hparams = parse_args(parser)
    hparams.use_gpu = torch.cuda.is_available() and hparams.gpu >= 1

    if hparams.hparams_file and os.path.exists(hparams.hparams_file):
        print(f"Loading hparams from {hparams.hparams_file} ...")
        hparams.load(hparams.hparams_file)
        print(f"Loaded hparams from {hparams.hparams_file}")
    print(json.dumps(hparams, indent=2))
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)
    hparams.save(os.path.join(hparams.save_dir, "hparams.json"))

    def set_seed(seed):
        """ fix random seed """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if hparams.gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def to_tensor(array):
        """
        numpy array -> tensor
        """
        if isinstance(array, np.ndarray):
            array = torch.tensor(array)
            return array.cuda() if hparams.use_gpu else array
        else:
            return array

    # set seed
    set_seed(seed=hparams.seed)

    # set reader
    bpe = IntentBPETextField(hparams)
    hparams.Model.num_token_embeddings = bpe.vocab_size
    hparams.Model.num_turn_embeddings = bpe.max_ctx_turn + 1

    # set data paths and collate function
    labeled_data_paths = bpe.labeled_data_paths
    unlabeled_data_paths = bpe.unlabeled_data_paths
    collate_fn = bpe.collate_fn_multi_turn

    # loading datasets and score matrixs
    train_label_loader, valid_label_loader, test_label_loader = None, None, None
    train_nolabel_loader, valid_nolabel_loader, test_nolabel_loader = None, None, None
    if hparams.do_train:
        train_label_loader = get_sequential_data_loader(batch_size=hparams.batch_size_label, reader=bpe,
                                                        hparams=hparams, data_paths=labeled_data_paths,
                                                        collate_fn=collate_fn, data_type='train')
        bpe.load_score_matrix(data_type='train', data_iter=train_label_loader)

        if hparams.learning_method == 'semi':
            train_nolabel_loader = get_sequential_data_loader(batch_size=hparams.batch_size_nolabel, reader=bpe,
                                                              hparams=hparams, data_paths=unlabeled_data_paths,
                                                              collate_fn=collate_fn, data_type='train')
    if hparams.do_test:
        valid_label_loader = get_sequential_data_loader(batch_size=hparams.batch_size_label, reader=bpe,
                                                        hparams=hparams, data_paths=labeled_data_paths,
                                                        collate_fn=collate_fn, data_type='valid')
        bpe.load_score_matrix(data_type='valid', data_iter=valid_label_loader)

        if hparams.learning_method == 'semi':
            valid_nolabel_loader = get_sequential_data_loader(batch_size=hparams.batch_size_nolabel, reader=bpe,
                                                              hparams=hparams, data_paths=unlabeled_data_paths,
                                                              collate_fn=collate_fn, data_type='valid')
    if hparams.do_infer:
        test_label_loader = get_sequential_data_loader(batch_size=hparams.batch_size_label, reader=bpe,
                                                       hparams=hparams, data_paths=labeled_data_paths,
                                                       collate_fn=collate_fn, data_type='test')
        bpe.load_score_matrix(data_type='test', data_iter=test_label_loader)

        if hparams.learning_method == 'semi':
            test_nolabel_loader = get_sequential_data_loader(batch_size=hparams.batch_size_nolabel, reader=bpe,
                                                             hparams=hparams, data_paths=unlabeled_data_paths,
                                                             collate_fn=collate_fn, data_type='test')

    # construct generator
    generator = Generator.create(hparams, reader=bpe)

    # construct model
    model = ModelBase.create(hparams, reader=bpe, generator=generator)
    print("Total number of parameters in networks is {}".format(sum(x.numel() for x in model.parameters())))

    # multi-gpu setting
    if hparams.gpu > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # construct trainer
    trainer = IntentTrainer(model, to_tensor, hparams, reader=bpe)

    # set optimizer and lr_scheduler
    if hparams.do_train:
        if hparams.learning_method == 'semi':
            num_batches = max(len(train_label_loader), len(train_nolabel_loader))
        else:
            num_batches = len(train_label_loader)
        trainer.set_optimizers(num_training_steps_per_epoch=num_batches)

    # load model, optimizer and lr_scheduler
    trainer.load()

    # training process
    if hparams.do_train:
        trainer.train(train_label_iter=train_label_loader, train_nolabel_iter=train_nolabel_loader,
                      valid_label_iter=valid_label_loader, valid_nolabel_iter=valid_nolabel_loader)

    # inference process
    if hparams.do_infer and not hparams.do_train:
        train_label_loader = get_sequential_data_loader(batch_size=hparams.batch_size_label, reader=bpe,
                                                        hparams=hparams, data_paths=labeled_data_paths,
                                                        collate_fn=collate_fn, data_type='train')
        trainer.infer(data_iter=test_label_loader, ex_data_iter=train_label_loader)


if __name__ == "__main__":
    main()
