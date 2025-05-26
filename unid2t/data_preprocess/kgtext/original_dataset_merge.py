import sys
import os

from data_preprocess.utils import read_json_file, write_to_json_file


def merge(file_prefix, output_file_src):
    dataset_types = ['train.json', 'val.json', 'test.json']

    datasets = []

    for dataset_type in dataset_types:
        file_src = os.path.join(file_prefix, dataset_type)

        sub_dataset = read_json_file(file_src)
        datasets.extend(sub_dataset)

        print("{} dataset has {} examples".format(dataset_type, len(sub_dataset)))

    write_to_json_file(datasets, output_file_src)
    print("Finished merge, total {} examples have been saved at: {}".format(len(datasets), output_file_src))


if __name__ == '__main__':
    original_data_dir = '../../original_datasets/kgtext/'
    merge_file_src = '../../original_datasets/kgtext/merged.json'
    merge(original_data_dir, merge_file_src)
