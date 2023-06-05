import json
import pdb

# from map_subword_serialize_sampling import schema_subword_matrix
from map_subword_serialize import schema_subword_matrix
import argparse
from transformers import AutoTokenizer
import pickle

def schema_subword_dataset(seq2seq_dataset, tokenizer, tables, output_path = None):
    i = 0
    for i_str, data in seq2seq_dataset.items():
        table_items = data['db_table_names']
        column_items = data['db_column_names']['column_name']
        # sampled_columns_idx = data['sampled_columns_idx']
        # new_mapping = data['new_mapping'] # TODO

        db_id = data['db_id']
        # db_sep = data['struct_in']
        db_sep = data['serialized_schema']
        # db_sep = 'schema: {}'.format(data['struct_in'])   
        schema_relations = tables[db_id]
        subword_matrix, subword_mapping_dict, new_struct_in, schema_to_ids = schema_subword_matrix(db_sep=db_sep, table_items=table_items, tokenizer=tokenizer,
                                                                    column_items=column_items, init_idx=0, tables=tables)
                                                                    # column_items=column_items, new_mapping=new_mapping, init_idx=0, tables=tables)
                                                                    # column_items=column_items, init_idx=0, tables=tables, sampled_columns_idx=sampled_columns_idx)
        data['schema_subword_relations'] = subword_matrix           
        data['schema_relations'] = schema_relations
        data['schema_subword_mapping_dict'] = subword_mapping_dict
        data['new_struct_in'] = new_struct_in
        data['schema_to_ids'] = schema_to_ids
        # data['schema_idx_ori'] = schema_idx_ori
        # data['sampled_columns_idx'] = sampled_columns_idx
        i += 1
        if i % 1000 == 0:
            print("******************************* processing {}th datasets *******************************".format(i))


    if output_path:
        pickle.dump(seq2seq_dataset, open(output_path, "wb"))

    return seq2seq_dataset


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=False, help='dataset path')
    arg_parser.add_argument('--table_path', type=str, required=False, help='table path')
    arg_parser.add_argument('--dataset_output_path', type=str, required=False, help='dataset_output_path')
    arg_parser.add_argument('--schema_output_path', type=str, required=False, help='schema_output_path')
    arg_parser.add_argument('--plm_name', type=str, required=True, help='plm path')
    args = arg_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
    tokenizer.add_tokens([' <', ' <='])

    tables = pickle.load(open(args.table_path, "rb"))
    seq2seq_dataset = pickle.load(open(args.dataset_path, "rb"))
    new_tables = schema_subword_dataset(seq2seq_dataset=seq2seq_dataset, tokenizer=tokenizer, tables=tables,
                                        output_path=args.dataset_output_path)

    print(" schema subword construction finished ")