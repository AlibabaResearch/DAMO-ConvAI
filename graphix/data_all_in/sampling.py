import json
import pickle
import pdb
import argparse

dummy_relations = ['question-table-nomatch', 'question-column-nomatch', 'column-question-nomatch', 'table-question-nomatch']

def flatten_fk(foreign_keys_lst):
    final_lst = []
    for fk_pairs in foreign_keys_lst:
        for columns in fk_pairs:
            final_lst.append(columns)
    return list(set(final_lst))

def mapping_idx(sampled_columns_idx):
    mapping_dict = {}
    
    for i, col_idx in enumerate(sampled_columns_idx):
        mapping_dict[col_idx] = i
    return mapping_dict

def recaption_fk(ori_fk_lsts, sampled_columns_idx_mapping):
    new_fk_lsts = []

    for ori_fk_lst in ori_fk_lsts:
        new_fk_lsts.append([sampled_columns_idx_mapping[col] for col in ori_fk_lst])
    
    return new_fk_lsts
    

def sampling_database(dataset, tables, output_path=None):

    for idx, data in enumerate(dataset):
        sampled_columns_idx = []
        schema_linking = data['schema_linking']
        db_corr = tables[data['db_id']]
        table_names_original = db_corr['table_names_original']
        column_names_original = db_corr['column_names_original']
        primary_keys = db_corr['primary_keys']
        foreign_keys = db_corr['foreign_keys']
        table_len = len(table_names_original)
        # make sure 
        assert len(schema_linking[-1]) == len(db_corr['relations'])
        # keep the original database structures:
        # primary keys + foreign keys:
        fk_idx = flatten_fk(foreign_keys)
        pk_idx = primary_keys
        sampled_columns_idx.sort()
        sampled_columns_idx = list(set(fk_idx + pk_idx))
        for i in range(len(schema_linking[0])):
            for j in range(len(schema_linking[1])):
                if schema_linking[0][i][j] not in dummy_relations and j >= table_len:
                    sampled_columns_idx.append(j-table_len)
        sampled_columns_idx.sort()
        sampled_columns_idx = list(set(sampled_columns_idx))
        sampled_columns_idx_mapping = mapping_idx(sampled_columns_idx)

        # recaption foreign keys and primary keys:
        
        new_primary_keys = []
        for pk in primary_keys:
            if pk not in sampled_columns_idx_mapping:
                pdb.set_trace()
            new_primary_keys.append(sampled_columns_idx_mapping[pk])
        new_primary_keys = new_primary_keys
        new_foreign_keys = recaption_fk(foreign_keys, sampled_columns_idx_mapping)
        assert len(primary_keys) == len(new_primary_keys)
        assert len(foreign_keys) == len(new_foreign_keys)

        # collection:
        new_column_names_original = []
        for col_idx in sampled_columns_idx:
            new_column_names_original.append(column_names_original[col_idx])
        data['new_column_names_original'] = new_column_names_original
        data['new_table_names_original'] = table_names_original
        data['new_primary_keys'] = new_primary_keys
        data['new_foreign_keys'] = new_foreign_keys
        new_column_types = []
        for co_idx1 in sampled_columns_idx:
            new_column_types.append(db_corr['column_types'][co_idx1])

        # double check:

        assert len(new_column_names_original) == len(sampled_columns_idx)
        data['new_column_types'] = new_column_types
        data['graph_idx_eval'] = idx
        data['sampled_columns_idx'] = sampled_columns_idx
    
    if output_path:
        json.dump(dataset, open(output_path, 'w'), indent=4)

    return dataset 


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--tables_dir', type=str,)
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    arg_parser.add_argument('--skip_large', action='store_true', help='whether skip large databases')
    arg_parser.add_argument('--verbose', action='store_true', help='whether print processing information')
    args = arg_parser.parse_args()

    dev = pickle.load(open(args.dataset_path, "rb"))
    databases = pickle.load(open(args.tables_dir, "rb"))

    sampled_database = sampling_database(dataset=dev, tables=databases, output_path=args.output_path)

