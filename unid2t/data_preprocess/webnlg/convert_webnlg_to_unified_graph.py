import sys
import os
import copy
import argparse
import ast
from datasets import load_dataset
from data_preprocess.utils import write_to_json_file_by_line


def prediction_word_process(colloquial_word):
    # if colloquial_word in ['By', 'In', 'Of', 'At', 'On']:
    #     colloquial_word = colloquial_word.lower()
    if colloquial_word[0].isupper() and colloquial_word[1:].islower():
        colloquial_word = colloquial_word.lower()
    return colloquial_word


def make_prediction_colloquial(p):
    word = " ".join(p.split('_'))
    colloquial_words = []

    tmp_char = []
    is_char_small = False
    in_block = 0
    for i, char in enumerate(word):
        if char in ["'"]:
            in_block += 1
        elif in_block > 0 and char in ["'"]:
            in_block -= 1

        if in_block == 0:
            if char.islower():
                is_char_small = True
            elif char.isupper() and is_char_small:
                colloquial_word = "".join(tmp_char)
                colloquial_word = prediction_word_process(colloquial_word)
                colloquial_words.append(colloquial_word)
                tmp_char = []
                is_char_small = False
                # char = char.lower()
        tmp_char.append(char)

    if len(tmp_char):
        colloquial_word = "".join(tmp_char)
        colloquial_word = prediction_word_process(colloquial_word)
        colloquial_words.append(colloquial_word)

    colloquial_words = " ".join(colloquial_words)
    colloquial_words = " ".join(colloquial_words.split())
    return colloquial_words

"""
def colloquial_prediction(p):
    # was given the 'Technical Campus' status
    # was given the ' technical campus' status
    p = " ".join(p.split('_'))
    colloquial_p = make_prediction_colloquial(p)

    return colloquial_p
"""

def node_process(node, node_type='s', is_colloquial=False):

    if is_colloquial:
        if node_type == 's':
            # if '_' in node:
            #     print("s", node)
            # print("before s", node)
            node = " ".join(node.split('_'))
            # print("after  s", node)
        elif node_type == 'p':
            # print("before p", node)
            node = make_prediction_colloquial(node)
            # print("after  p", node)
        else:
            # print("o", node)
            node = " ".join(node.split('_'))

    return node


def triple_process(modified_triple_sets, is_simplified_data=True, is_colloquial=False):
    """

    :param {'mtriple_set': [['Aarhus_Airport | cityServed | "Aarhus, Denmark"']]}
    :param is_simplified_data:
    :param is_colloquial:
    :return:
    """
    mtriple_set = modified_triple_sets['mtriple_set']
    assert len(mtriple_set) == 1
    mtriple_set = mtriple_set[0]
    linear_nodes = []
    triples = []
    # print("mtriple_set", mtriple_set)
    for ori_triple in mtriple_set:
        s, p, o = ori_triple.split(' | ')

        s = node_process(s, node_type='s', is_colloquial=is_colloquial)
        p = node_process(p, node_type='p', is_colloquial=is_colloquial)
        o = node_process(o, node_type='o', is_colloquial=is_colloquial)

        if is_simplified_data:
            if s not in linear_nodes:
                s_idx = len(linear_nodes)
                linear_nodes.append(s)
            else:
                s_idx = linear_nodes.index(s)

            if p not in linear_nodes:
                p_idx = len(linear_nodes)
                linear_nodes.append(p)
            else:
                p_idx = linear_nodes.index(p)

            if o not in linear_nodes:
                o_idx = len(linear_nodes)
                linear_nodes.append(o)
            else:
                o_idx = linear_nodes.index(o)

        else:
            s_idx = len(linear_nodes)
            linear_nodes.append(s)

            p_idx = len(linear_nodes)
            linear_nodes.append(p)

            o_idx = len(linear_nodes)
            linear_nodes.append(o)


        triples.append((s_idx, 'f', p_idx))
        triples.append((p_idx, 'f', o_idx))

    return linear_nodes, triples


def one_example_process(example, is_simplified_data=True, is_colloquial=False):
    """

    :param example: dict_keys(['category',
    'size',
    'eid',
    'original_triple_sets',
    'modified_triple_sets',
    'shape',
    'shape_type',
    'lex',
    'test_category',
    'dbpedia_links',
    'links'])
    :param is_simplified_data
    :param is_colloquial
    :return:
    """
    processed_examples = []
    modified_triple_sets = example['modified_triple_sets']
    # for k, v in example.items():
    #     print(k, v)
    # print(example.keys())
    texts = example['lex']['text']

    linear_nodes, triples = triple_process(modified_triple_sets,
                                           is_simplified_data=is_simplified_data, is_colloquial=is_colloquial)
    for text in texts:

        processed_example = copy.deepcopy(example)
        processed_example['linear_node'] = linear_nodes
        processed_example['triple'] = triples
        processed_example['target_sents'] = [text]
        processed_example['metadata'] = None

        processed_examples.append(processed_example)

    return processed_examples


def sub_dataset_process(dataset, dataset_type, output_file_dir, is_simplified_data=True, is_colloquial=False):

    dataset_size = len(dataset)

    processed_examples = []
    for i in range(dataset_size):

        example = dataset[i]
        tmp_processed_examples = one_example_process(example, is_simplified_data=is_simplified_data, is_colloquial=is_colloquial)

        processed_examples.extend(tmp_processed_examples)

    print("-- Finished, Original {} data-to-text pairs, final {} processed examples".format(dataset_size, len(processed_examples)))

    output_file_src = os.path.join(output_file_dir, "{}.json".format(dataset_type))
    write_to_json_file_by_line(processed_examples, output_file_src)
    print("-- {} dataset has been saved at: {}".format(dataset_type, output_file_src))


def process(output_dir, dataset_path='web_nlg', dataset_version='release_v2.1',
            is_simplified_data=True, is_colloquial=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # path = '/Users/liliang/WorkPlace/Datasets/web_nlg'
    # path = 'web_nlg'
    # name = 'release_v2.1'

    web_nlg = load_dataset(path=dataset_path, name=dataset_version)

    for dataset_type, sub_dataset in web_nlg.items():
        print("- Begin process {}: {}".format(dataset_version, dataset_type))
        sub_dataset_process(sub_dataset, dataset_type, output_dir,
                            is_simplified_data=is_simplified_data, is_colloquial=is_colloquial)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert webnlg to unified graph')

    parser.add_argument("--output_dir", type=str, default='../../cleanout_datasets/cleanout_webnlg/'
                                                          'webnlg_challenge_2017_with_unified_graph_simplified_colloquial')
    parser.add_argument("--dataset_path", type=str, default='web_nlg')
    parser.add_argument("--dataset_version", type=str, default='webnlg_challenge_2017',
                        help='webnlg_challenge_2017, release_v1, release_v2, release_v2.1')
    parser.add_argument("--is_simplified_data", type=ast.literal_eval, default=True)
    parser.add_argument("--is_colloquial", type=ast.literal_eval, default=True)

    args = parser.parse_args()
    process(output_dir=args.output_dir, dataset_path=args.dataset_path,
            dataset_version=args.dataset_version,
            is_simplified_data=args.is_simplified_data, is_colloquial=args.is_colloquial)


