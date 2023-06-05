import json
import pdb

# from map_subword_serialize_sampling import schema_linking_subword_sampled
from map_subword_serialize import schema_linking_subword
import argparse
from transformers import AutoTokenizer
import pickle

def schema_linking_subword_dataset(seq2seq_dataset, output_path = None):
    i = 0
    for i_str, data in seq2seq_dataset.items():
        question_subword_dict = data['question_subword_dict']
        schema_to_ids = data['schema_to_ids']
        question_subword_len = len(data['question_subword_matrix'])
        schema_subword_len = len(data['schema_subword_relations'])
        schema_linking = data['schema_linking']
        # schema_idx_ori = data['schema_idx_ori']
        # sampled_columns_idx = data['sampled_columns_idx']
        
        
        # new_mapping_zip = (data['new_mapping'], data['db_table_names'], data['db_column_names']['column_name']) # TODO

        schema_linking_subwords = schema_linking_subword(
            question_subword_dict=question_subword_dict,
            schema_2_ids=schema_to_ids,
            question_subword_len=question_subword_len,
            schema_subword_len=schema_subword_len,
            schema_linking=schema_linking,
            # new_mapping_zip=new_mapping_zip # TODO
        )
        
        # schema_linking_subwords = schema_linking_subword_sampled(
        #     question_subword_dict=question_subword_dict,
        #     schema_2_ids=schema_to_ids,
        #     question_subword_len=question_subword_len,
        #     schema_subword_len=schema_subword_len,
        #     schema_linking=schema_linking,
        #     schema_idx_ori=schema_idx_ori,
        #     # new_mapping_zip=new_mapping_zip # TODO
        # )

        data['schema_linking_subword'] = schema_linking_subwords
        i += 1

        if i % 1000 == 0:
            print("******************************* processing {}th datasets *******************************".format(i))

    if output_path:
        pickle.dump(seq2seq_dataset, open(output_path, "wb"))

    return seq2seq_dataset

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--dataset_output_path', type=str, required=False, help='dataset_output_path')
    arg_parser.add_argument('--plm_name', type=str, required=False, help='plm path')
    args = arg_parser.parse_args()

    seq2seq_dataset = pickle.load(open(args.dataset_path, "rb"))
    new_dataset = schema_linking_subword_dataset(seq2seq_dataset=seq2seq_dataset, output_path=args.dataset_output_path)

    print("finished schema-linking preprocessing")

