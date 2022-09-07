"""
Dataset class
"""

import json


class Dataset(object):
    """ Basic Dataset interface class. """

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Dataset")
        group.add_argument("--data_dir", type=str, required=False,
                           help="The dataset dir.")
        group.add_argument("--data_name", type=str, required=True,
                           choices=["uniDAunDial", "camrest", "kvret", "multiwoz"],
                           help="The name of dataset.")
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

    def __init__(self, data_file, hparams, transform=lambda s: json.loads(s)):
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
        self.world_size = hparams.world_size
        self.global_rank = hparams.global_rank
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
        return self.transform(self.fp.readline().strip())
