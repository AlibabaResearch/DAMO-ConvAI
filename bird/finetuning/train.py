import logging
import os
import time

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from collections import OrderedDict
import utils.tool
from utils.configue import Configure
from utils.dataset import TokenizedDataset
from utils.trainer import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
import pickle
import pdb

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)

graph_pedia = pickle.load(open('graph_pedia_total.bin', 'rb'))

def compute_params(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_num: {}, trainable_num:{}".format(total_num, trainable_num))
    return 

def main() -> None:
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # torch.set_deterministic(True)
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    # from filelock import FileLock
    # import nltk
    # with FileLock(".lock") as lock:
    #     nltk.download("punkt", quiet=True)
    #     nltk.download("stopwords", quiet=True)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)

    if 'checkpoint-???' in args.bert.location:
        args.bert.location = get_last_checkpoint(
            os.path.dirname(args.bert.location.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {args.bert.location.model_name_or_path}")

    if "wandb" in training_args.report_to and training_args.local_rank <= 0:
        import wandb

        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "uni-frame-for-knowledge-tabular-tasks"),
            name=training_args.run_name,
            # entity=os.getenv("WANDB_ENTITY", 'sgtnew'),
            **init_args,
        )
        wandb.config.update(training_args, allow_val_change=True)

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    # The inputs will be train, dev, test or train, dev now.
    # We deprecate the k-fold cross-valid function since it causes too many avoidable troubles.

    if not args.arg_paths:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                                                                         cache_dir=args.dataset.data_store_path)
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
            raw_datasets_split, cache_root)
    else:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        meta_tuning_data = {}
        for task, arg_path in args.arg_paths:
            task_args = Configure.Get(arg_path)
            task_args.bert = args.bert
            print('task_args.bert.location:', task_args.bert.location)
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
                path=task_args.dataset.loader_path,
                cache_dir=task_args.dataset.data_store_path)
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
                to_seq2seq(task_raw_datasets_split, cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(meta_tuning_data)

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    model = utils.tool.get_model(args.model.name)(args)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    # pdb.set_trace()
    compute_params(model)
    model_tokenizer = model.tokenizer
    

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")

    # We wrap the "string" seq2seq data into "tokenized tensor".
    train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                     seq2seq_train_dataset, graph_pedia) if seq2seq_train_dataset else None
    eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_eval_dataset, graph_pedia) if seq2seq_eval_dataset else None
    test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset, graph_pedia) if seq2seq_test_dataset else None
    # pdb.set_trace()
    # Initialize our Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 5)
    trainer = EvaluateFriendlySeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=model_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
        wandb_run_dir=wandb.run.dir if "wandb" in training_args.report_to and training_args.local_rank <= 0 else None,
        callbacks=[early_stopping_callback],
    )
    print('Trainer build successfully.')

    # Load model weights (for --do_train=False or post finetuning).
    if training_args.load_weights_from:
        state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)
        # release memory
        del state_dict

    if args.load_multiple_prefix_module_weights_from:
        reconstruct_state_dict = OrderedDict()

        # load prefix modules
        for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from:
            state_dict = torch.load(os.path.join(module_weight_location, transformers.WEIGHTS_NAME), map_location="cpu")
            MULTI_PREFIX_ATTR_NAME = "multi_prefix"
            for weight_name, stored_tensor in state_dict.items():
                if str(weight_name).startswith("pretrain_model"):
                    continue  # skip the pretrained model and we will load a new one from another place
                reconstruct_state_dict['{}.{}.{}'.format(MULTI_PREFIX_ATTR_NAME, "_".join(task_name.split("_")[:-1]), weight_name)] = stored_tensor
                # extract the prefix part and add them to dict

        # give it into the model
        trainer.model.load_state_dict(reconstruct_state_dict, strict=False)

        # release memory
        del reconstruct_state_dict

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="eval"
        )
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=test_dataset if test_dataset else eval_dataset,
            test_examples=seq2seq_test_dataset if seq2seq_test_dataset else seq2seq_eval_dataset,
            metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        max_predict_samples = len(test_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()