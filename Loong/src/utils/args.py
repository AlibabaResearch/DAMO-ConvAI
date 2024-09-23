#!/user/bin/env python
# coding=utf-8
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='args for evaluate.py')
    parser.add_argument("--models", "-c", default="gpt4o.yaml")
    parser.add_argument("--eval_model", type=str, default="gpt4.yaml")
    parser.add_argument('--debug_num', type=int, default=15, help="Control the number of generated items. If <0, it means using all data")
    parser.add_argument('--shuffle_prompts', action="store_true")
    parser.add_argument('--debug_level', type=str, default="1,2,3,4", help="Represents the level to be evaluated, eg: 1,2 or 3")
    parser.add_argument('--debug_set', type=str, default="1,2,3,4", help="Represents the set level to be evaluated, eg: 1,2 or 3")
    parser.add_argument('--process_num_gen', type=int, default=10)
    parser.add_argument('--process_num_eval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1000000007)
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--doc_path', type=str, default='./doc')
    parser.add_argument('--input_path', type=str, default='../data/loong.jsonl')
    parser.add_argument('--output_process_path', type=str, default='../data/loong_process.jsonl')
    parser.add_argument('--output_path', type=str, default='../output/loong_generate.jsonl')
    parser.add_argument('--evaluate_output_path', type=str, default='../output/loong_evaluate.jsonl')
    parser.add_argument('--max_length', type=int, default=300000)
    parser.add_argument('--domain', type=str, default='', help='financial, paper, legal')
    parser.add_argument('--add_noise', action="store_true", help="A boolean flag that defaults to False")
    parser.add_argument('--rag', action="store_true", help="whether to use rag model")
    parser.add_argument('--rag_num', type=int, help="recall top n")
    parser.add_argument('--continue_gen', action="store_true", help="whether to continue_generate from exist file")
    parser.add_argument('--model_config_dir', type=str, default='../config/models')

    args = parser.parse_args()
    return args

