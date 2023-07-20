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
from space.data.dataset import Dataset
from space.data.fields.gen_field import BPETextField, MultiWOZBPETextField, CamRestBPETextField, KvretBPETextField, dst_MultiWOZBPETextField
from space.trainers.gen_trainer import Trainer, MultiWOZTrainer, CamRestTrainer, KvretTrainer
from space.models.model_base import ModelBase
from space.models.generator import Generator
from space.utils.eval import MultiWOZEvaluator, CamRestEvaluator, KvretEvaluator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", type=str2bool, default=False,
                        help="Whether to run trainning in plato setting.")
    parser.add_argument("--do_test", type=str2bool, default=False,
                        help="Whether to run evaluation on the test dataset.")
    parser.add_argument("--do_dst", type=str2bool, default=False,
                        help="Whether to run evaluation on the test dataset.")    
    parser.add_argument("--do_infer", type=str2bool, default=False,
                        help="Whether to run inference on the test dataset.")
    parser.add_argument("--num_infer_batches", type=int, default=None,
                        help="The number of batches need to infer.\n"
                             "Stay 'None': infer on entrie test dataset.")
    parser.add_argument("--hparams_file", type=str, default=None,
                        help="Loading hparams setting from file(.json format).")
    BPETextField.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)
    Trainer.add_cmdline_argument(parser)
    ModelBase.add_cmdline_argument(parser)
    Generator.add_cmdline_argument(parser)

    hparams = parse_args(parser)
    hparams.use_gpu = torch.cuda.is_available() and hparams.gpu >= 1

    print(json.dumps(hparams, indent=2))

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)
    hparams.save(os.path.join(hparams.save_dir, "hparams.json"))

    def to_tensor(array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array)
        return array.cuda() if hparams.use_gpu else array

    def set_seed(seed):
        """ fix random seed """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # set seed
    set_seed(seed=hparams.seed)

    # set reader and evaluator
    if hparams.data_name == 'camrest':
        bpe = CamRestBPETextField(hparams)
        evaluator = CamRestEvaluator(reader=bpe)
    elif hparams.data_name == 'multiwoz':
        if hparams.do_dst == True:
            print('using non-delex')
            bpe = dst_MultiWOZBPETextField(hparams)
            # print('using non-delex')
        else:
            bpe = MultiWOZBPETextField(hparams)
        evaluator = MultiWOZEvaluator(reader=bpe)
    elif hparams.data_name == 'kvret':
        bpe = KvretBPETextField(hparams)
        evaluator = KvretEval
        uator(reader=bpe)
    else:
        raise NotImplementedError("Other dataset's reader and evaluator to be implemented !")

    hparams.Model.num_token_embeddings = bpe.vocab_size
    hparams.Model.num_turn_embeddings = bpe.max_ctx_turn + 1

    # set data and data status
    if hparams.do_train:
        train_data = bpe.get_batches('train')
        dev_data = bpe.get_batches('dev')
        # print(len(train_data))
    else:
        train_data, dev_data, = [], []
    # print(train_data)
    # print(type(train_data))
    # print(train_data[0])
    # set generator
    generator = Generator.create(hparams, reader=bpe)

    # construct model
    model = ModelBase.create(hparams, reader=bpe, generator=generator)
    print("Total number of parameters in networks is {}".format(sum(x.numel() for x in model.parameters())))

    # multi-gpu
    if hparams.gpu > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # construct trainer
    if hparams.data_name == 'camrest':
        trainer = CamRestTrainer(model, to_tensor, hparams, reader=bpe, evaluator=evaluator)
    elif hparams.data_name == 'multiwoz':
        trainer = MultiWOZTrainer(model, to_tensor, hparams, reader=bpe, evaluator=evaluator)
    elif hparams.data_name == 'kvret':
        trainer = KvretTrainer(model, to_tensor, hparams, reader=bpe, evaluator=evaluator)
    else:
        raise NotImplementedError("Other dataset's trainer to be implemented !")

    # set optimizer and lr_scheduler
    if hparams.do_train:
        trainer.set_optimizers()

    # load model, optimizer and lr_scheduler
    trainer.load()

    if hparams.do_train:
        # training process
        trainer.train(train_data=train_data, dev_data=dev_data)

    if hparams.do_infer:
        # infer process
        if hparams.do_dst == True:
            trainer.infer_dst(data_type='test')
        else:
            trainer.infer(data_type='test')


if __name__ == "__main__":
    main()
