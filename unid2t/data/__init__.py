from transformers import T5Tokenizer

from data.totto.text_to_text_dataset import Text2TextDataset
from data.unified_dataset import UnifiedGraphDataset
from data.unified_Iterable_dataset import UnifiedIterableDataset


def init_dataset(data_dir, tokenizer: T5Tokenizer, special_tokens: list = [],
                 datatype=None, max_inp_len=-1, max_target_len=-1, n_example=-1,
                 enable_uda_relative_pos=False, data_processor='linear', task_source_prefix=None,
                 noise_processor=None, rank=0, num_gpus=1, dataset_style='map',position_style = "0"):
    if data_processor == 'totto_text2text':
        dataset = Text2TextDataset(
            data_dir=data_dir, datatype=datatype, tokenizer=tokenizer,
            special_tokens=special_tokens,
            max_inp_len=max_inp_len,
            max_target_len=max_target_len,
            n_example=n_example,
            task_source_prefix=task_source_prefix
        )

    else:
        if dataset_style == 'map':
            dataset = UnifiedGraphDataset(data_dir=data_dir,
                                          datatype=datatype,
                                          tokenizer=tokenizer,
                                          special_tokens=special_tokens,
                                          max_inp_len=max_inp_len,
                                          max_target_len=max_target_len,
                                          is_processed=False,
                                          n_example=n_example,
                                          enable_uda_relative_pos=enable_uda_relative_pos,
                                          data_processor=data_processor,
                                          task_source_prefix=task_source_prefix,
                                          noise_processor=noise_processor,
                                          position_style = position_style)
        else:  # iterable
            n_lines = sum(1 for line in open(data_dir, 'r'))
            n_lines = n_lines if n_example == -1 else min(n_lines, n_example)
            dataset = UnifiedIterableDataset(
                data_dir=data_dir, datatype=datatype, tokenizer=tokenizer,
                special_tokens=special_tokens,
                max_inp_len=max_inp_len,
                max_target_len=max_target_len,
                n_lines=n_lines, enable_uda_relative_pos=enable_uda_relative_pos,
                data_processor=data_processor, task_source_prefix=task_source_prefix,
                noise_processor=noise_processor, rank=rank, num_gpus=num_gpus,
            )

    return dataset
