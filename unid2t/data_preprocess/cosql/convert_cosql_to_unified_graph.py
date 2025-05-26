import sys
import os
import copy
import argparse
import ast
from tqdm import tqdm
from tools.logger import init_logger
from data_preprocess import utils


def convert_table_into_triple(table, results_map: dict, is_lower=True):
    if table is None:
        return ['none'], [(0, 'Col', 0)]
    if len(table) == 1:
        n_col = len(table[0])
        table.append(['none' for i in range(n_col)])

    table = [row for row in table if row is not None]

    linear_nodes = []
    node_ids = []
    triples = []

    # [Row] [Col]
    for row_idx, row in enumerate(table):
        ids = []
        for col_id, value in enumerate(row):
            if results_map is not None and value in results_map.keys():
                value = results_map[value]

            head_idx = len(linear_nodes)
            ids.append(head_idx)
            linear_nodes.append(value)

            # same column
            for pre_same_col_idx in range(col_id):

                previous_same_col_value = row[pre_same_col_idx]
                if results_map is not None and previous_same_col_value in results_map.keys():
                    previous_same_col_value = results_map[previous_same_col_value]
                assert previous_same_col_value in linear_nodes

                tail_idx = ids[pre_same_col_idx]

                triples.append((head_idx, '[Col]', tail_idx))

            # same row
            for pre_same_row_idx in range(row_idx):
                previous_same_row_value = table[pre_same_row_idx][col_id]
                if results_map is not None and previous_same_row_value in results_map.keys():
                    previous_same_row_value = results_map[previous_same_row_value]
                assert previous_same_row_value in linear_nodes

                tail_idx = node_ids[pre_same_row_idx][col_id]

                triples.append((head_idx, '[Row]', tail_idx))
        node_ids.append(ids)

    linear_nodes = [str(node) for node in linear_nodes]
    return linear_nodes, triples


def obtain_response(example: dict, is_lower=True):
    """

    :param example:
    :param is_lower:
    :return:
    """

    response = example.get('response_toks_blank_res', None)
    if response is None:
        assert example['results_map'] is None
        response = example.get('response_toks', None)
        assert response is not None
    if response is None:
        return None
    response = " ".join(response)
    if is_lower:
        response = response.lower()

    return response


def obtain_query(example: dict, is_lower=True):

    query = example.get('query_toks_w_results_blank_res', None)
    if query is None:
        assert example['results_map'] is None, example
        query = example.get('query_toks', None)
        assert query is not None

    query = " ".join(query)
    if is_lower:
        query = query.lower()

    return query


def convert_one_example(example: dict, is_lower=True):
    """
    :param example:
        database_id: str
        query: str
        query_toks: list(str)
        results_raw: list(list())
        response: str
        response_toks: str
        results_map: dict
        results_blank_res: str
        query_w_results: str
        query_w_results_blank_res: str
        query_toks_w_results: list(str)
        query_toks_w_results_blank_res: list(str)
        response_blank_res: str
        response_toks_blank_res: list(str)
    :param is_lower
    """
    # print(example)
    database_id = example['database_id']
    results_raw = example['results_raw']
    results_map = example['results_map']

    # if results_raw is None:
    #     print(example)
    target_sent = obtain_response(example, is_lower=is_lower)
    # assert target_sent is not None
    if target_sent is None:
        return None
    # print("target_sent", target_sent)

    query = obtain_query(example, is_lower)
    linear_nodes, triples = convert_table_into_triple(table=results_raw, results_map=results_map, is_lower=is_lower)

    converted_example = dict()

    converted_example['linear_node'] = linear_nodes
    converted_example['triple'] = triples
    converted_example['target_sents'] = [target_sent]
    converted_example['metadata'] = [query]

    converted_example['database_id'] = database_id
    converted_example['results_raw'] = results_raw
    converted_example['results_map'] = results_map

    return converted_example


def convert(input_file_src, output_file_src, is_lower):
    logger = init_logger(__name__)
    dataset = utils.read_json_file(input_file_src)

    n_examples = len(dataset)

    converted_dataset = []
    n_nodes = 0
    n_enc_tokens = 0
    n_dec_tokens = 0
    n_triples = 0

    targets = []

    for example in tqdm(dataset):

        converted_example = convert_one_example(example, is_lower)
        if converted_example is None:
            continue

        targets.append(converted_example['target_sents'][0])

        n_nodes += len(converted_example['linear_node'])
        n_triples += len(converted_example['triple'])
        n_enc_tokens += len(" ".join(converted_example['linear_node']).split())
        n_dec_tokens += len(converted_example['target_sents'][0].split())

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

    utils.write_to_txt_file(targets, output_file_src.replace('.json', '_only_predict.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wikitablet to unified graph')

    parser.add_argument("--input_file_src", type=str,
                        default='../../orig_dataset/cosql/cosql_train.json')
    parser.add_argument("--output_file_src", type=str,
                        default='../../cleanout_dataset/cosql_with_unified_graph/cosql_train.json')
    parser.add_argument("--is_lower", type=ast.literal_eval, default=True)

    args = parser.parse_args()

    convert(input_file_src=args.input_file_src, output_file_src=args.output_file_src,
            is_lower=args.is_lower)






