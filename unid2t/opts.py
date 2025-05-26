import argparse
import ast
import yaml
import json

import random
import numpy as np
import torch

from tools.logger import init_logger

logger = init_logger(__name__)


def set_seed(args):
    reproducibility = getattr(args, 'reproducibility', True)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def load_config_from_file(args):
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f_in:
            config_data = f_in.read()
            configs = yaml.safe_load(config_data)
            assert type(configs) == dict, type(configs)

            for arg, value in configs.items():
                setattr(args, arg, value)

    noise_task_source_prefix_str = getattr(args, 'noise_task_source_prefix', None)
    if noise_task_source_prefix_str is not None:
        noise_task_source_prefix = json.loads(noise_task_source_prefix_str)
        setattr(args, 'noise_task_source_prefix', noise_task_source_prefix)

    if getattr(args, 'plms_enable_sim_cse', False):
        plms_sim_ces_config_str = getattr(args, 'plms_sim_cse_config', None)
        if plms_sim_ces_config_str is not None:
            plms_sim_ces_config = json.loads(plms_sim_ces_config_str)
            setattr(args, 'plms_sim_cse_config', plms_sim_ces_config)

    if getattr(args, 'update_freq', 1) > 1:
        train_batch_size = int(args.train_batch_size // args.update_freq)
        logger.info("- Enable gradient accumulation, the train batch size is modified from "
                    "{} to {}".format(args.train_batch_size, train_batch_size))
        args.train_batch_size = train_batch_size

    return args


def print_args(args):
    logger.info('Input Argument Information')

    args_dict = {k: str(v) for k, v in vars(args).items()}
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))


def init_opts():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--config', required=False,
                        help='Path of the main YAML config file.')
    parser.add_argument('--save_config', required=False, default=None,
                        help='Path where to save the config.')

    # basic configs
    parser.add_argument("--reproducibility", type=ast.literal_eval, default=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--basic_model', default='t5', choices=['t5', 'bart'])
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--data_processor', default='linear', choices=['linear', 'uda', 'totto_text2text'])
    parser.add_argument('--model_name', type=str, default='uda', choices=['uda', 't5'])
    parser.add_argument('--datatype', type=str, default='graph', choices=['graph', 'linear'])

    # data
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--init_model_path", type=str)
    parser.add_argument("--special_token_path", type=str)
    parser.add_argument("--dataset_style", type=str, default='map', choices=['map', 'iterable'])

    # parser.add_argument("--save_tokenizer", default=None, help='Path where to save the tokenizer')

    # model
    parser.add_argument("--task_source_prefix", type=str, default=None)
    # for uda
    parser.add_argument("--enable_uda_relative_pos", type=ast.literal_eval, default=False)

    return parser


def init_opts_for_finetuning():
    parser = init_opts()
    model_opts(parser)
    training_opts(parser)
    evaluate_opts(parser)

    args = parser.parse_args()
    args = load_config_from_file(args)
    opts_check(args)
    if args.local_rank == 0:
        print_args(args)
    """
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    """
    set_seed(args)
    return args


def init_opts_for_pretraining():
    parser = init_opts()
    model_opts(parser)
    training_opts(parser)
    pretraining_opts(parser)
    evaluate_opts(parser)

    args = parser.parse_args()
    args = load_config_from_file(args)
    opts_check(args)

    if args.local_rank == 0:
        print_args(args)
    set_seed(args)

    return args


def init_opts_for_inference():
    parser = init_opts()
    model_opts(parser)
    inference_opts(parser)
    evaluate_opts(parser)

    args = parser.parse_args()
    args = load_config_from_file(args)

    print_args(args)
    set_seed(args)

    return args


def raw_dataset_process_opts(parser: argparse.ArgumentParser):
    row_data_parser = parser.add_argument_group("Raw Dataset Preprocess")

    row_data_parser.add_argument('--processes_num', type=int, default=1)


def training_opts(parser: argparse.ArgumentParser):
    training_parser = parser.add_argument_group("Training Configuration")

    # distributed training and speedup
    training_parser.add_argument("--dist_train", type=ast.literal_eval, default=False)
    training_parser.add_argument("--local_rank", type=int, default=0, help='node rank for distributed training')
    training_parser.add_argument("--distributed_env", type=str, default='distributed_data_parallel',
                                 choices=['distributed_data_parallel', 'speedup'])
    # general
    training_parser.add_argument("--train_type", type=str, default='finetune', choices=['pretrain', 'finetune'])
    training_parser.add_argument("--experiment_name", type=str)
    training_parser.add_argument("--max_epochs", type=int, default=30)
    training_parser.add_argument("--max_steps", type=int, default=-1)
    training_parser.add_argument("--early_stopping_patience", type=int, default=5)
    training_parser.add_argument("--start_eval_from", type=int, default=0)
    training_parser.add_argument("--eval_every", type=int, default=1)
    training_parser.add_argument("--max_keep_checkpoints", type=int, default=10)
    training_parser.add_argument("--report_every", type=int, default=100)
    training_parser.add_argument("--saved_dir", type=str)
    training_parser.add_argument("--update_freq", type=int, default=1, help='gradient accumulation')

    # train
    training_parser.add_argument("--learner", type=str, default='adamw', choices=['adamw', 'adam', 'adagrad',
                                                                                  'adafactor', 'fairseq_adafactor'],
                                 help="adamw: 1e-4, 5e-5; adam: ; adagrad: ,"
                                      "adafactor: 5e-4; fairseq_adafactor: 1e-3")
    training_parser.add_argument("--learning_rate", type=float, default=1e-4)
    training_parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    training_parser.add_argument("--max_grad_norm", type=float, default=2.0)
    training_parser.add_argument("--lr_scheduler", type=str, default='none', choices=['none', 'linear', 'inverse_square'])
    training_parser.add_argument("--warmup_steps", type=int, default=0)

    # data
    training_parser.add_argument("--train_file_src", type=str)
    training_parser.add_argument("--n_train_example", type=int, default=-1)
    training_parser.add_argument("--train_batch_size", type=int, default=16)
    training_parser.add_argument("--max_source_length", type=int, default=128, help="Sequence length of instances.")
    training_parser.add_argument("--max_target_length", type=int, default=128, help="Target sequence length of instances.")
    training_parser.add_argument("--train_num_workers", type=int, default=5)
    training_parser.add_argument("--train_pin_memory", type=ast.literal_eval, default=False)

    return training_parser


def pretraining_opts(parser: argparse.ArgumentParser):
    pretraining_parser = parser.add_argument_group("Pretraining Configuration")

    # training
    pretraining_parser.add_argument("--random_delete_rate", type=float, default=0.15)
    pretraining_parser.add_argument("--noise_types", type=list, help='t5_denoising, data2text')
    pretraining_parser.add_argument("--noise_type_rates", type=list)
    pretraining_parser.add_argument("--noise_task_source_prefix", type=str, default=None,
                                    help="{'t5_denoising': 'Denoising the following data: ', "
                                         "'data2text': 'Describe the following data: '}")
    pretraining_parser.add_argument("--random_allocation_mask", type=ast.literal_eval, default=True,
                                    help='whether random allocation the mask id for the input nodes')

    # evaluating
    pretraining_parser.add_argument("--eval_noise_data", type=ast.literal_eval, default=False)
    pretraining_parser.add_argument("--eval_random_delete_rate", type=float, default=None)
    pretraining_parser.add_argument("--eval_noise_types", type=list, default=None)
    pretraining_parser.add_argument("--eval_noise_type_rates", type=list, default=None)
    pretraining_parser.add_argument("--eval_noise_task_source_prefix", type=str, default=None,
                                    help="{'t5_denoising': 'Denoising the following data: ', "
                                         "'data2text': 'Describe the following data: '}")


def evaluate_opts(parser: argparse.ArgumentParser):
    evaluate_parser = parser.add_argument_group("Evaluating Configuration")

    # general
    evaluate_parser.add_argument("--val_metric", type=str, default='bleu')
    evaluate_parser.add_argument("--eval_task_source_prefix", type=str, default=None)
    # data
    evaluate_parser.add_argument("--eval_file_src", type=str)
    evaluate_parser.add_argument("--n_eval_example", type=int, default=-1)
    evaluate_parser.add_argument("--eval_batch_size", type=int, default=16)
    evaluate_parser.add_argument("--num_beams", type=int, default=5)
    evaluate_parser.add_argument("--eval_max_source_length", type=int, default=-1)
    evaluate_parser.add_argument("--eval_max_target_length", type=int, default=None)
    evaluate_parser.add_argument("--eval_num_workers", type=int, default=5)

    return evaluate_parser


def inference_opts(parser: argparse.ArgumentParser):
    inference_parser = parser.add_argument_group("Inference Configuration")

    inference_parser.add_argument("--checkpoint_path", type=str)
    inference_parser.add_argument("--infer_generated_text_dir", type=str)
    inference_parser.add_argument("--file_save_prefix", type=str, default=None)


def model_opts(parser: argparse.ArgumentParser):

    model_parser = parser.add_argument_group("Model Configuration")

    model_parser.add_argument("--modified_default_plm_config", type=ast.literal_eval, default=False)
    model_parser.add_argument("--plms_dropout_rate", type=float, default=0.1)
    # SimCSE config
    model_parser.add_argument("--plms_enable_sim_cse", default=False, type=ast.literal_eval)
    model_parser.add_argument("--plms_sim_cse_config", default=None, type=str,
                              help='{"alpha_sim_cse": 1.0, "sim_cse_temp": 0.05, "pooler_type": "avg"}')


def opts_check(args):
    # training
    if args.max_epochs > 0:
        assert args.max_steps == -1
    else:
        assert args.max_steps > 0





