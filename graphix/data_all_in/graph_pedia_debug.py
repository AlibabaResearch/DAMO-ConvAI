import json
import pdb

# from map_subword_serialize_sampling import schema_linking_subword_sampled
from map_subword_serialize import schema_linking_subword
import argparse
from transformers import AutoTokenizer
import pickle

def merge_graph_pedia(graph_pedia_train, graph_pedia_dev, graph_all_output_path=None):
    # keep the index of train set as original.
    graph_pedia_all = pickle.load(open(graph_pedia_train, "rb"))
    dev_start_idx = len(graph_pedia_all)
    graph_pedia_dev = pickle.load(open(graph_pedia_dev, "rb"))
    
    for dev_idx, v in graph_pedia_dev.items():
        graph_pedia_all[dev_idx + dev_start_idx] = v
    
    pickle.dump(graph_pedia_all, open(graph_all_output_path, 'wb'))
    return graph_pedia_dev

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--graph_train_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--graph_dev_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--graph_all_output_path', type=str, required=False, help='graph_all_output_path')
    args = arg_parser.parse_args()

    merge_graph = merge_graph_pedia(graph_pedia_train=args.graph_train_path, graph_pedia_dev=args.graph_dev_path, graph_all_output_path=args.graph_all_output_path)

    print("finished merging graph_pedia")

