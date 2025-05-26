import math
from copy import deepcopy
import numpy as np
import torch

from tools.logger import init_logger


class NoiseProcessor(object):
    def __init__(self, logger, extra_tokens: list, random_delete_rate: float = 0.15,
                 noise_types: list = ['t5_denoising'], noise_type_rates: list = [1.0],
                 noise_task_source_prefix: dict = None, random_allocation_mask=True):

        self.logger = logger
        self.extra_tokens = extra_tokens
        self.random_delete_rate = random_delete_rate
        self.noise_types = noise_types
        self.noise_type_rates = noise_type_rates
        self.noise_task_source_prefix = noise_task_source_prefix
        self.random_allocation_mask = random_allocation_mask
        if noise_task_source_prefix is not None:
            assert list(noise_task_source_prefix.keys()) == noise_types, "{}: {}".format(noise_task_source_prefix.keys(), noise_types)

    @classmethod
    def init_noise_processor(cls, extra_tokens, args=None, random_delete_rate=None, noise_types=None,
                             noise_type_rates=None, noise_task_source_prefix=None, random_allocation_mask=True):
        logger = init_logger(__name__)
        data_type = 'train'
        if random_delete_rate is None:
            random_delete_rate = args.random_delete_rate
            data_type = 'val'
        if noise_types is None:
            noise_types = args.noise_types
        if noise_type_rates is None:
            noise_type_rates = args.noise_type_rates

        if noise_task_source_prefix is None:
            noise_task_source_prefix = args.noise_task_source_prefix

        if noise_type_rates is None or len(noise_type_rates) == 0:
            n_noise_types = len(noise_types)
            noise_type_rates = [1 / n_noise_types for i in range(n_noise_types)]

        if args:
            random_allocation_mask = args.random_allocation_mask

        logger.info("Enable noise processor for {}, noise types include: {}, type rates are: {}".format(data_type,
                                                                                                        noise_types,
                                                                                                        noise_type_rates))
        if noise_task_source_prefix is not None:
            logger.info("Add noise task source prefix: {}".format(noise_task_source_prefix))
        logger.info("Random_allocation_mask set as: {}".format(random_allocation_mask))
        return cls(logger=logger, extra_tokens=extra_tokens, random_delete_rate=random_delete_rate,
                   noise_types=noise_types, noise_type_rates=noise_type_rates,
                   noise_task_source_prefix=noise_task_source_prefix, random_allocation_mask=random_allocation_mask)

    def inject_noising(self, example: dict):
        if example['target_sents'] is None:
            noise_type = 't5_denoising'
        else:
            
            noise_type = np.random.choice(self.noise_types, 1,
                                          p=self.noise_type_rates,
                                          replace=False)
            noise_type = noise_type.tolist()[0]
        # self.logger.info("noise_type: {}: {}".format(noise_type,type(noise_type)))
        if noise_type == 't5_denoising':
            example = self.t5_denoise_processor(example=example)

        elif noise_type == 'data2text':
            return example, noise_type
        else:
            raise NotImplementedError

        return example, noise_type

    def t5_denoise_processor(self, example: dict):
        """

        :param example: dict(linear_node, triple, metadata, target_sents)
        :param extra_tokens: ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>', ..., '<extra_id_99>']
        :param random_delete_rate:
        :return:
        """

        linear_node = example['linear_node']
        n_node = len(linear_node)
        n_random_delete = math.ceil(len(linear_node) * self.random_delete_rate)

        n_random_delete = n_random_delete if n_random_delete <= 100 else 100

        deleted_node_indices = np.random.choice(a=n_node, size=n_random_delete, replace=False)
        # print("init deleted_node_indices", deleted_node_indices)
        if not self.random_allocation_mask:
            deleted_node_indices.sort()
        # print("after deleted_node_indices", deleted_node_indices)
        target = []
        corruption_nodes = deepcopy(linear_node)
        for i, deleted_node_idx in enumerate(deleted_node_indices):
            # print("{} -- {}".format(i, deleted_node_idx))
            replace_token = self.extra_tokens[i]
            # dropped_out_token = corruption_nodes[i]
            dropped_out_token = linear_node[deleted_node_idx]
            # print("replace_token", replace_token)
            # print("dropped_out_token", dropped_out_token)
            target.append(replace_token)
            target.append(dropped_out_token)
            corruption_nodes[deleted_node_idx] = replace_token

        assert len(corruption_nodes) == n_node
        # corruption_nodes = " ".join(corruption_nodes)
        # print("corruption_nodes", corruption_nodes)
        # print("init target", target)
        target = " ".join(target)
        example['linear_node'] = corruption_nodes
        example['target_sents'] = [target]

        return example
