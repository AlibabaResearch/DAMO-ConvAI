"""
Preprocess script.
"""

import os
import argparse

from galaxy.args import parse_args
from galaxy.data.dataset import Dataset
from galaxy.data.pretrain_field import PretrainBPETextField
from galaxy.models.model_base import ModelBase

LABELED_TAG = 0
UNLABELED_TAG = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_file", type=str, required=True,
                        help="The name of labeled dataset (.json format).")
    parser.add_argument("--unlabeled_file", type=str, required=True,
                        help="The name of unlabeled dataset (.json format).")
    PretrainBPETextField.add_cmdline_argument(parser)
    ModelBase.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)

    args = parse_args(parser)

    bpe = PretrainBPETextField(args)
    build_examples_fn = bpe.build_examples_multi_turn

    labeled_train_file = os.path.join(args.data_dir, f'{args.labeled_file}.json')
    assert os.path.exists(labeled_train_file), f"{labeled_train_file} isn't exist"
    print(f'Assign tag={LABELED_TAG} to {labeled_train_file}')

    unlabeled_train_file = os.path.join(args.data_dir, f'{args.unlabeled_file}.json')
    assert os.path.exists(unlabeled_train_file), f"{unlabeled_train_file} isn't exist"
    print(f'Assign tag={UNLABELED_TAG} to {unlabeled_train_file}')

    train_file = os.path.join(args.data_dir, f'{args.data_name}.{args.tokenizer_type}.jsonl')

    if not os.path.exists(train_file):
        labeled_train_examples = build_examples_fn(labeled_train_file, data_type="train", tag=LABELED_TAG)
        unlabeled_train_examples = build_examples_fn(unlabeled_train_file, data_type="train", tag=UNLABELED_TAG)
        train_examples = labeled_train_examples + unlabeled_train_examples
        bpe.save_examples(train_examples, train_file)
    else:
        print(f'{train_file} already exists!')


if __name__ == "__main__":
    main()
