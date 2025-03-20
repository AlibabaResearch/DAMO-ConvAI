import json
import sys
import os
import numpy as np
import argparse
import ast

sys.path.append('/UnifiedData2TextPretrain')
from tools.logger import init_logger


class FileProcessor(object):
    def __init__(self):
        self.logger = init_logger(__name__)
        self.examples = []
        self.val_examples = []

    def read_dataset(self, file_src):
        with open(file_src, 'r') as f_in:
            lines = f_in.readlines()
            lines = [line.strip() for line in lines]
            self.examples.extend(lines)

        self.logger.info("Loading {} examples from {}".format(len(lines), file_src))

    def write_dataset(self, output_file_src):
        with open(output_file_src, 'w') as f_out:
            for example in self.examples:
                f_out.write(example + '\n')

        self.logger.info("Finished, {} examples have been saved at {}".format(self.dataset_size, output_file_src))
        if len(self.val_examples):
            val_output_file_src = output_file_src.replace('.json', '_val.json')
            with open(val_output_file_src, 'w') as f_out:
                for example in self.val_examples:
                    f_out.write(example + '\n')
            self.logger.info("Finished, {} validation examples have been saved at {}".format(len(self.val_examples),
                                                                                             val_output_file_src))

    def shuffle_data(self):
        self.logger.info("Beginning to shuffle {} examples".format(self.dataset_size))
        data_ids = np.arange(self.dataset_size)
        self.logger.info("init data_ids: {}".format(data_ids))
        np.random.shuffle(data_ids)
        data_ids.tolist()
        self.logger.info("shuffled data_ids: {}".format(data_ids))
        new_dataset = [self.examples[example_idx] for example_idx in data_ids]
        self.examples = new_dataset

    def merged_dataset(self, file_src):
        # self.logger.info("begin me")
        if os.path.isdir(file_src):
            file_names = os.listdir(file_src)
            file_names = [file_name for file_name in file_names if file_name.endswith('.json') or file_name.endswith('.jsonl')]
            dataset_srcs = [os.path.join(file_src, file_name) for file_name in file_names]
            self.logger.info("Beginning to merge {} files".format(len(dataset_srcs)))
            for dataset_src in dataset_srcs:
                self.read_dataset(dataset_src)
        else:
            self.read_dataset(file_src)

        self.logger.info("Loading total {} examples".format(self.dataset_size))

    def split_out_val(self, val_size=1000):

        ids_with_target = []
        for i, str_example in enumerate(self.examples):
            example = json.loads((str_example))
            target = example['target_sents']
            if target is None or len(target) == 0:
                continue
            ids_with_target.append(i)

        np.random.shuffle(ids_with_target)
        val_ids = ids_with_target[:val_size]
        self.logger.info("Val Ids: {}".format(val_ids))
        train_examples = []
        val_examples = []
        for i, example in enumerate(self.examples):
            if i in val_ids:
                val_examples.append(example)
            else:
                train_examples.append(example)

        self.examples = train_examples
        self.val_examples = val_examples

    @property
    def dataset_size(self):
        return len(self.examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File merge and shuffle Configuration')
    parser.add_argument("--inp", type=str, help='the input file or dir path of input datasets')
    parser.add_argument("--merge", default=True, type=ast.literal_eval)
    parser.add_argument("--shuffle", default=True, type=ast.literal_eval)
    parser.add_argument("--out", type=str, help='the output file path of output dataset')
    parser.add_argument("--split_out_val_size", type=int, default=-1)

    args = parser.parse_args()

    processor = FileProcessor()
    if args.merge:
        processor.merged_dataset(args.inp)
    else:
        processor.read_dataset(args.inp)

    if args.split_out_val_size > 0:
        processor.split_out_val(val_size=args.split_out_val_size)

    if args.shuffle:
        processor.shuffle_data()

    if processor.dataset_size > 0:
        processor.write_dataset(args.out)
    else:
        processor.logger.info("The dataset size is 0")





