from map_function import question_subword_matrix
from transformers import AutoTokenizer
import pickle
import argparse

def question_subword_dataset(dataset, tokenizer, output_path=None):

    for i, data in enumerate(dataset):
        processed_question_toks = data['raw_question_toks']
        relations = data['relations']
        data['subword_relations'], data['subword_dict'] = question_subword_matrix(processed_question_toks=processed_question_toks,
                                                                                   relations=relations, tokenizer=tokenizer)


        if i % 500 == 0:
            print("processing {}th data".format(i))

    if output_path:
        pickle.dump(dataset, open(output_path, "wb"))
    return dataset

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--table_path', type=str, required=False, help='table path')
    arg_parser.add_argument('--dataset_output_path', type=str, required=False, help='dataset_output_path')
    arg_parser.add_argument('--schema_output_path', type=str, required=False, help='schema_output_path')
    arg_parser.add_argument('--plm_name', type=str, required=True, help='plm path')
    args = arg_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name, use_fast=True)
    tokenizer.add_tokens([' <', ' <='])

    dataset = pickle.load(open(args.dataset_path, "rb"))
    new_dataset = question_subword_dataset(dataset=dataset, tokenizer=tokenizer, output_path=args.dataset_output_path)