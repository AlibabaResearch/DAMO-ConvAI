#!/usr/bin/env python
# -*- coding:utf-8 -*-
from universal_ie.utils import label_format
import yaml
import os
from typing import Dict
import universal_ie.task_format as task_format


class Dataset:
    def __init__(self, name: str, path: str, data_class: task_format.TaskFormat, split_dict: Dict, language: str, mapper: Dict, other: Dict = None) -> None:
        self.name = name
        self.path = path
        self.data_class = data_class
        self.split_dict = split_dict
        self.language = language
        self.mapper = mapper
        self.other = other

    def load_dataset(self):
        datasets = {}
        for split_name, filename in self.split_dict.items():
            datasets[split_name] = self.data_class.load_from_file(
                filename=os.path.join(self.path, filename),
                language=self.language,
                **self.other,
            )
        return datasets

    @staticmethod
    def load_yaml_file(yaml_file):
        dataset_config = yaml.load(open(yaml_file), Loader=yaml.FullLoader)
        if 'mapper' in dataset_config:
            mapper = dataset_config['mapper']
            for key in mapper:
                mapper[key] = label_format(mapper[key])
        else:
            print(f"{dataset_config['name']} without label mapper.")
            mapper = None

        return Dataset(
            name=dataset_config['name'],  # 数据集名字 Name of Dataset
            path=dataset_config['path'],  # 数据集路径 Path of Dataset
            data_class=getattr(task_format, dataset_config['data_class']),  # 数据集对应的 Task Format 名字 Raw data loader
            split_dict=dataset_config['split'],   # 数据集不同划分文件地址 Data Split Path
            language=dataset_config['language'],  # 数据集语言 Dataset Language
            mapper=mapper,
            other=dataset_config.get('other', {}),
        )
