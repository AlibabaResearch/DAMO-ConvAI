#coding=utf8
import os, json, pickle, argparse, sys, time
import pdb
import math, dgl, torch
import numpy as np
import os, sys
from collections import defaultdict
from transformers import AutoTokenizer

# mapping special column * as an ordinary column
special_column_mapping_dict = {
    'question-*-generic': 'question-column-star',
    '*-question-generic': 'column-question-star',
    'table-*-generic': 'table-column-has',
    '*-table-generic': 'column-table-has',
    '*-column-generic': 'column-column-star',
    'column-*-generic': 'column-column-star',
    '*-*-identity': 'column-column-identity',
    'question-question-mod': 'question-question-modifier',
    'question-question-arg': 'question-question-argument',
    'question-question-dist1': 'question-question-dist-1',
}
nonlocal_relations = [
    'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
    'question-question-identity', 'table-table-identity',  'symbol', '0', '*-*-identity',
    'column-column-identity', 'question-table-nomatch', 'question-column-nomatch', 'column-question-nomatch', 'table-question-nomatch']

class SubwordGraphProcessor():

    def preprocess_subgraph(self, seq2seq_dataset: dict, concat_relation: list):
        graph = defaultdict()
        num_nodes = int(math.sqrt(len(concat_relation)))
        edges = [(idx // num_nodes, idx % num_nodes,(special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
                 for idx, r in enumerate(concat_relation) if r not in nonlocal_relations]
        src_ids, dst_ids = list(map(lambda r: r[0], edges)), list(map(lambda r: r[1], edges))
        graph['graph'] = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph['edges'] = edges

        '''construct mask for subword graph'''
        q_num = len(seq2seq_dataset['schema_linking_subword'][0])
        s_num = num_nodes - q_num
        graph['question_mask'] = [1] * q_num + [0] * s_num
        graph['schema_mask'] = [0] * q_num + [1] * s_num

        '''markdown indexes for subword_ids in graphs'''
        graph['question_subword_dict'] = seq2seq_dataset['question_subword_dict']
        graph['schema_to_ids'] = seq2seq_dataset['schema_to_ids']
        graph['len_question_subwords'] = len(seq2seq_dataset['question_subword_matrix'])
        graph['len_schema_subwords'] = len(seq2seq_dataset['schema_subword_relations'])

        joint_schema2ids = self.lag_subword_idx(len(seq2seq_dataset['question_subword_matrix']), seq2seq_dataset['schema_to_ids'])
        graph['joint_schema2ids'] = joint_schema2ids
        graph['node_idx'] = self.flatten_node_idx(seq2seq_dataset['question_subword_dict'], joint_schema2ids)
        graph['new_struct_in'] = seq2seq_dataset['new_struct_in']

        seq2seq_dataset['graph'] = graph
        # pdb.set_trace()

        return seq2seq_dataset

    def lag_subword_idx(self, add_idx, subword_dict):
        lagged_subword_dict = {}
        for k,v in subword_dict.items():
            lagged_subword_dict[k] = [v_i + add_idx for v_i in v]

        return lagged_subword_dict

    def flatten_node_idx(self, question_subword_dict: dict, schema_2_ids: dict):
        node_seq = []
        for k_q, v_q in question_subword_dict.items():
            node_seq.extend(v_q)

        for k_s, v_s in schema_2_ids.items():
            node_seq.extend(v_s)

        return node_seq


    def process_subgraph_utils(self, seq2seq_dataset):
        q = np.array(seq2seq_dataset['question_subword_matrix'], dtype='<U100')
        s = np.array(seq2seq_dataset['schema_subword_relations'], dtype='<U100')
        q_s = np.array(seq2seq_dataset['schema_linking_subword'][0], dtype='<U100')
        s_q = np.array(seq2seq_dataset['schema_linking_subword'][1], dtype='<U100')
        relation = np.concatenate([
            np.concatenate([q, q_s], axis=1),
            np.concatenate([s_q, s], axis=1)
        ], axis=0)
        relation = relation.flatten().tolist()

        seq2seq_dataset = self.preprocess_subgraph(seq2seq_dataset, relation)
        return seq2seq_dataset

def process_subgraph_datasets(processer, seq2seq_dataset, output_path = None, graph_output_path = None, graph_pedia=None, train_len=None):
    seq2seq_dataset_formal = []
    graph_pedia = defaultdict()

    for i_str, data in seq2seq_dataset.items():
        new_data = processer.process_subgraph_utils(data)
        graph_pedia[int(i_str)] = data['graph']

        # if graph_pedia is not None:
        #    graph_pedia[int(i_str) + train_len] = data['graph']


        del new_data['question_subword_matrix']
        del new_data['question_subword_dict']
        del new_data['question_token_relations']
        del new_data['schema_linking']
        del new_data['schema_subword_relations']
        del new_data['schema_relations']
        del new_data['schema_subword_mapping_dict']
        del new_data['schema_to_ids']
        del new_data['schema_linking_subword']
        del new_data['graph']
        seq2seq_dataset_formal.append(data)

        if int(i_str) % 1000 == 0:
            print("processing {}th data".format(int(i_str)))

    if output_path:
        json.dump(seq2seq_dataset_formal, open(output_path, "w"))

    if graph_output_path:
        pickle.dump(graph_pedia, open(graph_output_path, "wb"))


    return seq2seq_dataset_formal, graph_pedia

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--graph_pedia', type=str, required=False, help='output preprocessed dataset')
    arg_parser.add_argument('--output_path', type=str, required=False, help='output preprocessed dataset')
    arg_parser.add_argument('--graph_output_path', type=str, required=False, help='graph output preprocessed dataset')

    args = arg_parser.parse_args()

    processor = SubwordGraphProcessor()
    if args.graph_pedia:
        graph_pedia = pickle.load(open(args.graph_pedia, 'rb'))
    # pdb.set_trace()
    # loading database and dataset
    seq2seq_dataset = pickle.load(open(args.dataset_path, 'rb'))
    start_time = time.time()
    dataset = process_subgraph_datasets(
        processer=processor,
        seq2seq_dataset=seq2seq_dataset,
        output_path=args.output_path,
        graph_output_path=args.graph_output_path,
        # graph_pedia=graph_pedia,
        # train_len=8577
    )
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
    # pdb.set_trace()
    print('')








