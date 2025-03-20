import sys
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer

sys.path.append('/data/nt12_ssd_gluster/myself/experiment/UnifiedData2TextPretrain/')
from data_preprocess.utils import read_text_file

from data.unified_dataset import UnifiedGraphDataset


def statistic(dataset: UnifiedGraphDataset, save_prefix=None):
    data_size = len(dataset)

    n_enc_tokens = []
    n_enc_nodes = []
    n_dec_tokens = []

    for i in tqdm(range(data_size)):
        example = dataset.__getitem__(i)
        enc_inp = example['enc_inp']
        linear_node = example['linear_node']

        dec_inp = example['dec_inp']

        n_enc_tokens.append(len(enc_inp))
        n_enc_nodes.append(len(linear_node))
        n_dec_tokens.append(len(dec_inp))

    avg_n_enc_token = sum(n_enc_tokens) / data_size
    max_n_enc_token = max(n_enc_tokens)
    avg_n_enc_node = sum(n_enc_nodes) / data_size
    max_n_enc_node = max(n_enc_nodes)

    avg_n_dec_token = sum(n_dec_tokens) / data_size
    max_n_dec_token = max(n_dec_tokens)

    threshold_value = 512
    big_examples = [1 if n > threshold_value else 0 for n in n_enc_tokens]
    n_big_example = sum(big_examples)
    print("Total {} examples, avg_n_enc_token: {}, max_n_enc_token: {},"
          "avg_n_enc_node: {}, max_n_enc_node: {},"
          "avg_n_dec_token: {}, max_n_dec_token: {}".format(data_size, avg_n_enc_token, max_n_enc_token,
                                                            avg_n_enc_node, max_n_enc_node,
                                                            avg_n_dec_token, max_n_dec_token))

    print("There are {} examples have more than {} input tokens".format(n_big_example, threshold_value))
    if save_prefix is not None:
        np.save(file=save_prefix + '/n_enc_tokens.npy', arr=n_enc_tokens)
        np.save(file=save_prefix + '/n_enc_nodes.npy', arr=n_enc_nodes)
        np.save(file=save_prefix + '/n_dec_tokens.npy', arr=n_dec_tokens)


if __name__ == '__main__':
    # tokenizer_path = '/data/nt12_ssd_gluster/myself/pretrained_models/t5-base'
    tokenizer_path = '/Users/liliang/WorkPlace/pretrained_models/t5-small'
    special_token_path = '../cleanout_datasets/special_tokens.txt'
    # data_path = sys.argv[1]
    # save_prefix = sys.argv[2]
    data_path = '/Users/liliang/WorkPlace/Datasets/web_nlg/cleanout_webnlg/' \
                'web_nlg_release_v2.1_with_unified_graph_simplified_colloquial/test.json'
    save_prefix = None

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

    special_tokens = read_text_file(special_token_path)
    tokenizer.add_tokens(special_tokens)

    unified_dataset = UnifiedGraphDataset(
        data_dir=data_path, datatype=None, tokenizer=tokenizer, special_tokens=special_tokens,
        max_inp_len=-1, max_target_len=-1,
        is_processed=False, n_example=-1, enable_uda_relative_pos=False, data_processor='uda',
        task_source_prefix=None, noise_processor=None
    )

    statistic(unified_dataset, save_prefix=save_prefix)