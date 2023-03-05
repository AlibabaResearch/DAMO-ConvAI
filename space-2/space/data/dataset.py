"""
Dataset class
"""

import json
from space.args import str2bool


class Dataset(object):
    """ Basic Dataset interface class. """

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Dataset")
        group.add_argument("--data_dir", type=str, required=True,
                           help="The dataset dir.")
        group.add_argument("--with_mlm", type=str2bool, default=True,
                           help="Whether to use MLM loss and create MLM predictions.")
        group.add_argument("--with_contrastive", type=str2bool, default=True,
                           help="Whether to use contrastive loss and data augmentation.")
        group.add_argument("--num_process", type=int, default=1,
                           help="The num of processes to construct score matrix from respective dataset.")
        group.add_argument("--trigger_role", type=str, default="user",
                           choices=["user", "system", "user_system"], help="The modeling role side.")
        group.add_argument("--trigger_data", type=str, default="",
                           help="The name of triggered dataset for preprocessing and training.")
        group.add_argument("--dynamic_score", type=str2bool, default=True,
                           help="Whether to compute score matrix dynamically.")
        group.add_argument("--learning_method", type=str, default="super",
                           choices=["super", "semi"], help="The learning method")
        return group

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyDataset(Dataset):
    """
    Lazy load dataset from disk.

    Each line of data file is a preprocessed example.
    """

    def __init__(self, data_file, reader, transform=lambda s: json.loads(s)):
        """
        Initialize lazy dataset.

        By default, loading .jsonl format.

        :param data_file
        :type str

        :param transform
        :type callable
        """
        self.data_file = data_file
        self.transform = transform
        self.reader = reader
        self.offsets = [0]
        with open(data_file, "r", encoding="utf-8") as fp:
            while fp.readline() != "":
                self.offsets.append(fp.tell())
        self.offsets.pop()
        self.fp = open(data_file, "r", encoding="utf-8")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.fp.seek(self.offsets[idx], 0)
        sample = self.transform(self.fp.readline().strip())
        if self.reader.with_mlm:
            # sample = self.reader.create_masked_lm_predictions(sample)
            sample = self.reader.create_token_masked_lm_predictions(sample)
        return sample
