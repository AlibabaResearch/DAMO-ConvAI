import os
import math
from typing import Dict
from copy import deepcopy
import numpy as np
from datasets import DatasetDict
from random import shuffle
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataset import T_co

from utils.configue import Configure

"""
Meta-tuning concat part.
After we set up the datasets of different tasks, we need to concat them in certain order:
which may have some effect on performance.
And we also need to handle the trivial things, since for different data, we need to evaluate them in different ways.
"""


def upsample(data, weight):
    n_data = len(data)
    assert weight >= 1

    integral = list(range(n_data)) * int(math.floor(weight))
    residual = list(range(n_data))
    shuffle(residual)
    residual = residual[:int(n_data * (weight - int(math.floor(weight))))]
    return [deepcopy(data[idx]) for idx in integral + residual]


class MultiTaskWrapper(Dataset):
    def __init__(self, args_path2dataset, meta_args, section):
        if meta_args.load_multiple_prefix_module_weights_from:
            task_id2task_name = sorted(['_'.join(task_name.split('_')[:-1]) for task_name, module_weight_location in meta_args.load_multiple_prefix_module_weights_from])
            meta_args.task_id2task_name = task_id2task_name
            meta_args.task_name2task_id = {task_name: task_id for task_id, task_name in enumerate(task_id2task_name)}

        # Raw data and size.
        args_path2data = {}
        for args_path, dataset in args_path2dataset.items():
            args_path2data[args_path] = [dataset[idx] for idx in range(len(dataset))]

        # Up-weight.
        temp = meta_args.dataset.upsample_temp
        if temp and temp != 1 and section == 'train':
            # Dataset statistics.
            args_path2size = {}
            for args_path, data in args_path2data.items():
                args_path2size[args_path] = len(data)

            # Compute resampling weights.
            args_path2upsample = {}
            sum_tau_size = sum([np.exp(np.log(size) / temp) for size in args_path2size.values()])
            sum_size = sum(args_path2size.values())
            for args_path, size in args_path2size.items():
                tau_size = np.exp(np.log(size) / temp)
                args_path2upsample[args_path] = tau_size / sum_tau_size * sum_size / size

            # Compute upsampling weights.
            largest_args_path, _ = max(args_path2size.items(), key=lambda x: x[1])
            norm_coef = args_path2upsample[largest_args_path]
            for args_path in args_path2upsample.keys():
                args_path2upsample[args_path] = args_path2upsample[args_path] / norm_coef

            # Upsample.
            for args_path in sorted(args_path2data.keys()):
                args_path2data[args_path] = upsample(args_path2data[args_path], args_path2upsample[args_path])

            print('Before upsampling', args_path2size)
            print('Upsampling weights', args_path2upsample)
            print('After upsampling', {args_path: len(data) for args_path, data in args_path2data.items()})

        # Add description.
        if meta_args.model.use_description:
            for args_path, data in args_path2data.items():
                args = Configure.Get(args_path)
                description = args.model.description
                for item in data:
                    item['description'] = description

        # Add section and arg_path.
        for args_path, data in args_path2data.items():
            for item in data:
                item['section'] = section
                item['arg_path'] = args_path
                if meta_args.load_multiple_prefix_module_weights_from:
                    item['task_id'] = meta_args.task_name2task_id[os.path.basename(args_path)[:-len('.cfg')]]

        # Subset for dev.
        if section == 'dev' and meta_args.dataset.eval_num:
            for args_path in args_path2data.keys():
                full_data = args_path2data[args_path]
                eval_num = meta_args.dataset.eval_num
                if eval_num < len(full_data):
                    stride = 1.0 * len(full_data) / eval_num
                    args_path2data[args_path] = [full_data[int(idx * stride)] for idx in range(eval_num)]

        # Concatenate.
        self.dataset = []
        for args_path in sorted(args_path2data.keys()):
            self.dataset.extend(args_path2data[args_path])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class StrideWrapper(Dataset):
    def __init__(self, dataset, stride):
        self.dataset = dataset
        self.index2old_index = [idx * stride for idx in range(len(self.dataset) // stride)]

    def __getitem__(self, index):
        old_index = self.index2old_index[index]
        return self.dataset[old_index]

    def __len__(self):
        return len(self.index2old_index)


class DescriptionWrapper(Dataset):
    def __init__(self, dataset, description):
        self.dataset = dataset
        self.description = description

    def __getitem__(self, index):
        item = self.dataset[index]
        item['description'] = self.description
        return item

    def __len__(self):
        return len(self.dataset)


class SectionArgspathWrapper(Dataset):
    def __init__(self, dataset, section, args_path):
        self.dataset = dataset
        self.section = section
        self.args_path = args_path

    def __getitem__(self, index):
        item = self.dataset[index]
        item['section'] = self.section
        item['arg_path'] = self.args_path
        return item

    def __len__(self):
        return len(self.dataset)


class ConcatShuffleDataset(Dataset):
    def __init__(self, datasets):
        self.concat_dataset = ConcatDataset(datasets)
        self.index2old_index = list(range(len(self.concat_dataset)))
        np.random.shuffle(self.index2old_index)

    def __getitem__(self, index):
        old_index = self.index2old_index[index]
        return self.concat_dataset[old_index]

    def __len__(self):
        return len(self.concat_dataset)


class Constructor(object):
    def __init__(self, meta_args):
        self.meta_args = meta_args

    def to_seq2seq(self, raw_datasets_dict: Dict[str, DatasetDict]):
        """
        Construct the meta-tuning data for train, dev and test.
        @param raw_datasets_dict: Dict[arg_path, DatasetDict]
        @return:
        """
        train_dev_test_data_of_tasks = {'train': {}, "validation": {}, "test": {}}
        for arg_path, dataset in raw_datasets_dict.items():
            if len(dataset) == 2:
                train_dev_test_data_of_tasks['train'][arg_path] = dataset[0]
                train_dev_test_data_of_tasks['validation'][arg_path] = dataset[1]
                train_dev_test_data_of_tasks['test'][arg_path] = dataset[1]  # Use the dev set if no test set.
            elif len(dataset) == 3:
                train_dev_test_data_of_tasks['train'][arg_path] = dataset[0]
                train_dev_test_data_of_tasks['validation'][arg_path] = dataset[1]
                train_dev_test_data_of_tasks['test'][arg_path] = dataset[2]
            else:
                raise ValueError()

        train_dataset = TrainDataset(self.meta_args, train_dev_test_data_of_tasks['train'])
        dev_dataset = DevDataset(self.meta_args, train_dev_test_data_of_tasks['validation'])
        test_dataset = TestDataset(self.meta_args, train_dev_test_data_of_tasks['test'])

        return train_dataset, dev_dataset, test_dataset


class TrainDataset(Dataset):
    """
    Using the mata-tuning policy to control the data feeding order.
    """

    def __init__(self, meta_args, tasks_train_data: Dict[str, Dataset]):
        """
        DON'T shuffle the dataset. Please control it on parameter outside!!!
        @param meta_args: the meta args which control all the training.
        @param tasks_train_data:
        """
        self.meta_args = meta_args

        self.meta_training_data = MultiTaskWrapper(args_path2dataset=tasks_train_data, meta_args=meta_args, section='train')

    def __getitem__(self, index) -> T_co:
        return self.meta_training_data[index]

    def __len__(self):
        return len(self.meta_training_data)


class DevDataset(Dataset):
    """
    Add prefix info for evaluator to recognize.
    """

    def __init__(self, meta_args, tasks_dev_data):
        self.meta_args = meta_args

        self.meta_dev_data = MultiTaskWrapper(args_path2dataset=tasks_dev_data, meta_args=meta_args, section='dev')

    def __getitem__(self, index) -> T_co:
        return self.meta_dev_data[index]

    def __len__(self):
        return len(self.meta_dev_data)


class TestDataset(Dataset):
    """
    Add prefix info for evaluator to recognize.
    """

    def __init__(self, meta_args, tasks_test_data):
        self.meta_args = meta_args

        self.meta_test_data = MultiTaskWrapper(args_path2dataset=tasks_test_data, meta_args=meta_args, section='test')

    def __getitem__(self, index) -> T_co:
        return self.meta_test_data[index]

    def __len__(self):
        return len(self.meta_test_data)