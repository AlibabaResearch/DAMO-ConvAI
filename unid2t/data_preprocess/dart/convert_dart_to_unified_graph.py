import sys
import os
import copy
import argparse
import ast
from tqdm import tqdm
from tools.logger import init_logger
from data_preprocess import utils
from collections import Counter

cnt = Counter()


def relation_process(relation, is_lower_relation=True):
    if is_lower_relation:
        relation = relation.lower()
    relation = relation.split("_")
    relation = " ".join(relation)

    return relation


def linearize_tripleset(tripleset, is_simplified_data, is_lower_relation=True):
    """

    :param tripleset: [
            [head, relation, tail]
        ]
    :param is_simplified_data: if true: merge the same node
    :param is_lower_relation
    :return:
    """

    linear_nodes = []
    triples = []

    triple_dict = dict()
    for str_triple in tripleset:
        head, relation, tail = str_triple[0], str_triple[1], str_triple[2]

        relation = relation_process(relation, is_lower_relation=is_lower_relation)

        if head not in triple_dict.keys():
            triple_dict[head] = []
        triple_dict[head].append((relation, tail))

    for head, relation_tail_list in triple_dict.items():

        if is_simplified_data and head in linear_nodes:
            head_id = linear_nodes.index(head)
        else:
            head_id = len(linear_nodes)
            linear_nodes.append(head)

        for relation, tail in relation_tail_list:
            if is_simplified_data and relation in linear_nodes:
                relation_id = linear_nodes.index(relation)
            else:
                relation_id = len(linear_nodes)
                linear_nodes.append(relation)

            if is_simplified_data and tail in linear_nodes:
                tail_id = linear_nodes.index(tail)
            else:
                tail_id = len(linear_nodes)
                linear_nodes.append(tail)

            triples.append((head_id, relation_id))
            triples.append((relation_id, tail_id))

    return linear_nodes, triples


def convert_one_example(example, is_simplified_data=True, is_lower_relation=True):
    """

    :param example: dict{
        tripleset: [
            [head, relation, tail]
        ],
        subtree_was_extended: bool
        annotations: [dict{
            source: str,
            text: str
        }]
    }
    :return: converted_example: dict{
        tripleset: [
            [head, relation, tail]
        ],
        subtree_was_extended: bool

        linear_node: [str]
        triple: list([head_id, tail_id])
        metadata: []
        annotations_source: [source: str]
        target_sents: [text: str]


    }
    :param is_simplified_data
    :param is_lower_relation
    """
    tripleset = example['tripleset']
    subtree_was_extended = example.get('subtree_was_extended', None)
    annotations = example['annotations']

    metadata = []
    annotation_source = [annotation['source'] for annotation in annotations]
    target_sents = [annotation['text'] for annotation in annotations]
    cnt['exp_s'] += len(annotation_source)

    converted_example = dict()

    linear_nodes, triples = linearize_tripleset(tripleset, is_simplified_data=is_simplified_data,
                                                is_lower_relation=is_lower_relation)
    converted_example['linear_node'] = linear_nodes
    converted_example['triple'] = triples
    converted_example['metadata'] = metadata
    converted_example['annotations_source'] = annotation_source
    converted_example['target_sents'] = target_sents

    converted_example['tripleset'] = tripleset
    converted_example['subtree_was_extended'] = subtree_was_extended

    return converted_example


def convert(input_file_src, output_file_src, is_simplified_data=True, is_lower_relation=True):

    logger = init_logger(__name__)
    dataset = utils.read_json_file(input_file_src)
    n_examples = len(dataset)

    converted_dataset = []
    n_nodes = 0
    n_enc_tokens = 0
    n_dec_tokens = 0
    n_triples = 0

    for example in tqdm(dataset):
        cnt['exp'] += 1
        converted_example = convert_one_example(example, is_simplified_data=is_simplified_data,
                                                is_lower_relation=is_lower_relation)
        n_nodes += len(converted_example['linear_node'])
        n_triples += len(converted_example['triple'])
        n_enc_tokens += len(" ".join(converted_example['linear_node']).split())
        n_dec_tokens += len(converted_example['target_sents'][0].split())

        print('conver expL', len(converted_example['target_sents']), converted_example)

        converted_dataset.append(converted_example)

    n_converted_example = len(converted_dataset)
    utils.write_to_json_file_by_line(converted_dataset, output_file_src)
    logger.info("Finished, total {} examples, cleanout {} examples, "
                "avg_node: {}, avg_enc_tokens: {}, avg_dec_tokens: {}, avg_triples: {}, "
                "cleanout data has been saved at {}".format(n_examples, n_converted_example,
                                                            n_nodes / n_converted_example,
                                                            n_enc_tokens / n_converted_example,
                                                            n_dec_tokens / n_converted_example,
                                                            n_triples / n_converted_example,
                                                            output_file_src))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wikitablet to unified graph')

    parser.add_argument("--input_file_src", type=str,
                        default='../../orig_datasets/dart/data/v1.1.1/dart-v1.1.1-full-train.json')
    parser.add_argument("--output_file_src", type=str,
                        default='../../cleanout_datasets/dart/data/v1.1.1/dart_train.json')
    parser.add_argument("--is_simplified_data", type=ast.literal_eval, default=True)
    parser.add_argument("--is_lower_relation", type=ast.literal_eval, default=True)

    args = parser.parse_args()

    convert(input_file_src=args.input_file_src, output_file_src=args.output_file_src,
            is_simplified_data=args.is_simplified_data, is_lower_relation=args.is_lower_relation)

    print('cnt:', cnt)
