# -*- coding: utf-8 -*-

import argparse


def get_parser():
    """
    从命令行获取parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_dir", type=str, default="./output_dse_model/")
    parser.add_argument("--serving_dir", type=str, default="./serving_dse_model/")
    parser.add_argument("--model_name", type=str, default="dse")
    parser.add_argument("--train_file", type=str, default="./data/train.txt")
    parser.add_argument("--test_file", type=str, default="./data/test.txt")
    parser.add_argument("--env_name", type=str, default="dse")
    parser.add_argument("--load_step", type=str, default="1000")

    # 模型参数相关
    parser.add_argument("--do_lower", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="cl_bert")
    parser.add_argument("--bidirectional", type=str, default=1)

    # 数据相关的参数
    parser.add_argument("--train_multi_mode", type=bool, default=True)
    parser.add_argument("--test_multi_mode", type=bool, default=True)
    parser.add_argument("--line_sep_char", type=str, default="\t")
    parser.add_argument("--turn_sep_char", type=str, default="|")
    parser.add_argument("--token_sep_char", type=str, default="")
    parser.add_argument("--layer_num", type=int, default=6)
    parser.add_argument("--round_num", type=int, default=3)
    parser.add_argument("--class_num", type=int, default=2)
    parser.add_argument("--keep_prob", type=int, default=0.8)
    parser.add_argument("--train_pack_label", type=bool, default=False)
    parser.add_argument("--test_pack_label", type=bool, default=False)
    parser.add_argument("--train_response_num", type=int, default=10)
    parser.add_argument("--test_response_num", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--min_df", type=int, default=0)
    parser.add_argument("--column_num", type=int, default=-1)
    parser.add_argument("--partition_num", type=int, default=2)
    parser.add_argument("--load_batch", type=int, default=4)

    # 训练相关的参数
    parser.add_argument("--train_gpu", type=str, default="0,1,2,3")
    parser.add_argument("--test_gpu", type=str, default="0,1,2,3")
    parser.add_argument("--train_batch_size", type=int, default=20)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument("--epoches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--lr_decay_rate", type=float, default=0.98)
    parser.add_argument("--lr_warmup_steps", type=int, default=2000)
    parser.add_argument("--lr_decay_steps", type=int, default=10000)
    parser.add_argument("--print_per_steps", type=int, default=1)
    parser.add_argument("--eval_per_steps", type=int, default=10)
    parser.add_argument("--l2_reg", type=float, default=0.0003)

    # bert系列模型特有参数
    parser.add_argument("--use_init_model", type=bool, default=True)
    parser.add_argument("--bert_init_dir", type=str, default="./pretrain_model/") # download from BERT github
    parser.add_argument("--init_checkpoint", type=str, default="bert_model")
    parser.add_argument("--vocab_file", type=str, default="vocab.txt")
    parser.add_argument("--bert_config_file", type=str, default="bert_config.json")

    # 模式选择
    parser.add_argument("--stage", type=str, default="train")

    args = parser.parse_known_args()[0]
    return args

