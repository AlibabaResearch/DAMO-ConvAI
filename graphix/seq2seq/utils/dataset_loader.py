import json
from typing import Callable, Tuple
import logging
import datasets.load
from datasets.dataset_dict import DatasetDict
from datasets.metric import Metric
from datasets.arrow_dataset import Dataset, concatenate_datasets
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.training_args import TrainingArguments
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.dataset import (
    DataArguments,
    DataTrainingArguments,
    DatasetSplits,
    TrainSplit,
    _prepare_train_split,
    prepare_splits,
)
from seq2seq.utils.spider import spider_add_serialized_schema, spider_pre_process_function
from seq2seq.utils.cosql import cosql_add_serialized_schema, cosql_pre_process_function

logger = logging.getLogger(__name__)


def _log_duplicate_count(dataset: Dataset, dataset_name: str, split: str) -> None:
    d = dataset.to_dict()
    d_t = [tuple((k, tuple(v)) for k, v in zip(d.keys(), vs)) for vs in zip(*d.values())]
    d_t_ = set(d_t)
    num_examples = len(d_t)
    duplicate_count = num_examples - len(d_t_)
    if duplicate_count > 0:
        logger.warning(
            f"The split ``{split}`` of the dataset ``{dataset_name}`` contains {duplicate_count} duplicates out of {num_examples} examples"
        )


def load_dataset(
    data_args: DataArguments,
    model_args: ModelArguments,
    data_training_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Metric, DatasetSplits]:
    _spider_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["spider"], cache_dir=model_args.cache_dir
    )
    _spider_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["spider"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )
    _spider_add_serialized_schema = lambda ex: spider_add_serialized_schema(
        ex=ex,
        data_training_args=data_training_args,
    )
    _spider_pre_process_function = lambda batch, max_source_length, max_target_length: spider_pre_process_function(
        batch=batch,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )

    _cosql_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["cosql"], cache_dir=model_args.cache_dir
    )
    _cosql_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["cosql"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )
    _cosql_add_serialized_schema = lambda ex: cosql_add_serialized_schema(
        ex=ex,
        data_training_args=data_training_args,
    )
    _cosql_pre_process_function = lambda batch, max_source_length, max_target_length: cosql_pre_process_function(
        batch=batch,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )
    #adding spider_realistic dataset, metric, using schema and preprocess funtions of spider as it is
    _spider_realistic_dataset_dict : Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths['spider_realistic'], cache_dir=model_args.cache_dir
    )
    _spider_realistic_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["spider_realistic"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )

    _spider_syn_dataset_dict : Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths['spider_syn'], cache_dir=model_args.cache_dir
    )
    _spider_syn_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["spider_syn"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )

    _spider_dk_dataset_dict : Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths['spider_dk'], cache_dir=model_args.cache_dir
    )
    _spider_dk_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["spider_dk"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )

    _prepare_splits_kwargs = {
        "data_args": data_args,
        "training_args": training_args,
        "data_training_args": data_training_args,
    }

    if data_args.dataset == "spider":
        metric = _spider_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_spider_dataset_dict(),
            add_serialized_schema=_spider_add_serialized_schema,
            pre_process_function=_spider_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "cosql":
        metric = _cosql_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_cosql_dataset_dict(),
            add_serialized_schema=_cosql_add_serialized_schema,
            pre_process_function=_cosql_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "spider_realistic":
        metric = _spider_realistic_metric()
        dataset_splits = prepare_splits(
            dataset_dict= _spider_realistic_dataset_dict(),
            add_serialized_schema=_spider_add_serialized_schema,
            pre_process_function=_spider_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "spider_dk":
        metric = _spider_dk_metric()
        dataset_splits = prepare_splits(
            dataset_dict= _spider_dk_dataset_dict(),
            add_serialized_schema=_spider_add_serialized_schema,
            pre_process_function=_spider_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "spider_syn":
        metric = _spider_syn_metric()
        dataset_splits = prepare_splits(
            dataset_dict= _spider_syn_dataset_dict(),
            add_serialized_schema=_spider_add_serialized_schema,
            pre_process_function=_spider_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "cosql+spider":
        metric = _cosql_metric()
        cosql_dataset_splits = prepare_splits(
            dataset_dict=_cosql_dataset_dict(),
            add_serialized_schema=_cosql_add_serialized_schema,
            pre_process_function=_cosql_pre_process_function,
            **_prepare_splits_kwargs,
        )
        spider_training_split = (
            _prepare_train_split(
                dataset=_spider_dataset_dict()["train"],
                data_training_args=data_training_args,
                add_serialized_schema=_spider_add_serialized_schema,
                pre_process_function=_spider_pre_process_function,
            )
            if training_args.do_train
            else None
        )
        if cosql_dataset_splits.train_split is None and spider_training_split is None:
            train_split = None
        elif cosql_dataset_splits.train_split is None:
            train_split = spider_training_split
        elif spider_training_split is None:
            train_split = cosql_dataset_splits.train_split
        else:
            dataset: Dataset = concatenate_datasets(
                dsets=[cosql_dataset_splits.train_split.dataset, spider_training_split.dataset]
            )
            train_split = TrainSplit(
                dataset=dataset,
                schemas={**spider_training_split.schemas, **cosql_dataset_splits.train_split.schemas},
            )
        schemas = {
            **cosql_dataset_splits.schemas,
            **(spider_training_split.schemas if spider_training_split is not None else {}),
        }
        dataset_splits = DatasetSplits(
            train_split=train_split,
            eval_split=cosql_dataset_splits.eval_split,
            test_splits=cosql_dataset_splits.test_splits,
            schemas=schemas,
        )
    else:
        raise NotImplementedError()

    if dataset_splits.train_split is not None:
        _log_duplicate_count(dataset=dataset_splits.train_split.dataset, dataset_name=data_args.dataset, split="train")
    if dataset_splits.eval_split is not None:
        _log_duplicate_count(dataset=dataset_splits.eval_split.dataset, dataset_name=data_args.dataset, split="eval")
    if dataset_splits.test_splits is not None:
        for section, split in dataset_splits.test_splits.items():
            _log_duplicate_count(dataset=split.dataset, dataset_name=data_args.dataset, split=section)

    return metric, dataset_splits
