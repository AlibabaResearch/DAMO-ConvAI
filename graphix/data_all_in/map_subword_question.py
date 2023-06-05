import json
import pdb

from map_subword_serialize import question_subword_matrix
import argparse
from transformers import AutoTokenizer
import pickle

def question_subword_dataset(seq2seq_dataset, syntax_dataset, tokenizer, output_path = None):

    for i_str, data in seq2seq_dataset.items():

        processed_question_toks = data['raw_question_toks']
        relations = syntax_dataset[int(i_str)]['relations']
        question_sub_matrix, question_subword_dict, = \
        question_subword_matrix(processed_question_toks=processed_question_toks, relations=relations, tokenizer=tokenizer)

        data['question_subword_matrix'], data['question_subword_dict'] = question_sub_matrix, question_subword_dict
        data['question_token_relations'] = relations
        data['schema_linking'] = syntax_dataset[int(i_str)]['schema_linking']
        data['graph_idx'] = syntax_dataset[int(i_str)]['graph_idx']

        if int(i_str) % 500 == 0:
            print("processing {}th data".format(int(i_str)))

    if output_path:
        pickle.dump(seq2seq_dataset, open(output_path, "wb"))
    return seq2seq_dataset

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--syntax_path', type=str, required=False, help='syntax_dataset path')
    arg_parser.add_argument('--dataset_output_path', type=str, required=False, help='dataset_output_path')
    arg_parser.add_argument('--schema_output_path', type=str, required=False, help='schema_output_path')
    arg_parser.add_argument('--plm_name', type=str, required=True, help='plm path')
    args = arg_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name, use_fast=True)
    tokenizer.add_tokens([' <', ' <='])

    syntax_dataset = json.load(open(args.syntax_path, "r"))
    seq2seq_dataset = json.load(open(args.dataset_path, "r"))
    new_dataset = question_subword_dataset(seq2seq_dataset=seq2seq_dataset, syntax_dataset=syntax_dataset, tokenizer=tokenizer, output_path=args.dataset_output_path)

    print("finished question preprocessing")