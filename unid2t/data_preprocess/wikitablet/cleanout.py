import os
import sys
import argparse
from tqdm import tqdm
from data_preprocess.utils import read_json_file_by_line, write_to_json_file_by_line


def de_bpe(bpe_sent: str):
    """

    :param bpe_sent: Antoine L@@ avo@@ isi@@ er
    :return: Antoine Lavoisier
    """
    # tokens = tokenized_sent.split()
    sent = bpe_sent.replace('@@ ', '')
    return sent


def convert_data_to_graph(data):
    """

    :param data: list(list(str))
    :return:
    """
    key_value_dict = dict()
    for line in data:
        assert len(line) == 2
        key = line[0]
        value = line[1]

        if key not in key_value_dict.keys():
            key_value_dict[key] = []
        key_value_dict[key].append(value)

    linear_nodes = []
    triples = []
    key_ids = []
    for key, values in key_value_dict.items():
        start_idx = len(linear_nodes)
        key_ids.append(start_idx)
        linear_nodes.append(key)
        for value in values:
            value_idx = len(linear_nodes)
            linear_nodes.append(value)

            triples.append((start_idx, value_idx))

    n_keys = len(key_ids)
    for i in range(n_keys - 1):  # build connection between all keys
        for j in range(i + 1, n_keys):
            triples.append((key_ids[i], key_ids[j]))

    return linear_nodes, triples


def clean_one_example(example: dict):
    """

    :param example: {
                "doc_title": str,
                "doc_title_bpe": str,
                "sec_title": list(str),
                "data": list(list(str)
                "text": str
                "tokenized_text": list(str)
                }
    :return:
    """

    metadata = []
    doc_title = example['doc_title']

    doc_title = "the document title is: " + doc_title
    metadata.append(doc_title)

    bpe_sec_title = example['sec_title']
    sec_title = [de_bpe(value) for value in bpe_sec_title]

    sec_title = 'the section title is: ' + " ".join(sec_title)
    metadata.append(sec_title)

    bpe_data = example['data']
    data = [[de_bpe(value) for value in line] for line in bpe_data]

    linear_nodes, triples = convert_data_to_graph(data)
    bpe_text = example['text']
    text = de_bpe(bpe_text)
    clean_out_example = dict()

    clean_out_example['linear_node'] = linear_nodes
    clean_out_example['triple'] = triples
    clean_out_example['metadata'] = metadata
    clean_out_example['target_sents'] = [text]
    clean_out_example['data'] = data

    return clean_out_example


def clean_out(inp_file_src, out_file_src):

    dataset = read_json_file_by_line(inp_file_src)

    n_examples = len(dataset)
    
    clean_out_dataset = []
    n_nodes = 0
    n_enc_tokens = 0
    n_dec_tokens = 0
    n_skip = 0
    n_triples = 0
    
    for example in tqdm(dataset):
        clean_out_example = clean_one_example(example)
        n_nodes += len(clean_out_example['linear_node'])
        n_enc_tokens += len(" ".join(clean_out_example['linear_node']).split())
        n_dec_tokens += len(clean_out_example['target_sents'][0].split())
        n_triples += len(clean_out_example['triple'])

        clean_out_dataset.append(clean_out_example)

    n_cleanout_example = len(clean_out_dataset)

    write_to_json_file_by_line(clean_out_dataset, out_file_src)

    print("Finished, total {} examples, cleanout {} examples, "
          "avg_node: {}, avg_enc_tokens: {}, avg_dec_tokens: {}, avg_triples: {}, "
          "cleanout data has been saved at {}".format(n_examples, n_cleanout_example,
                                                      n_nodes / n_cleanout_example,
                                                      n_enc_tokens / n_cleanout_example,
                                                      n_dec_tokens / n_cleanout_example,
                                                      n_triples / n_cleanout_example,
                                                      out_file_src))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wikitablet to unified graph')

    parser.add_argument("--input_file_src", type=str,
                        default='../../orig_datasets/wikiTableT/final_data/test.json')

    parser.add_argument("--output_file_src", type=str,
                        default='../../clean_datasets/wikiTableT/final_data/test_udt.json')

    args = parser.parse_args()

    clean_out(inp_file_src=args.input_file_src, out_file_src=args.output_file_src)