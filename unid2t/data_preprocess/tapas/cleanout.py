import sys

sys.path.append('/data/nt12_ssd_gluster/myself/experiment/UnifiedData2TextPretrain/')
import os
import re
import argparse
import ast

from tqdm import tqdm
from nltk.corpus import stopwords

from data_preprocess import utils
from opts import print_args
from tools.logger import init_logger

logger = init_logger(__name__)
stopwords = set(stopwords.words('english'))


def convert_table_to_graph(table):
    linear_nodes = []
    node_dict = {}
    triples = []
    init_table_header = table['column']
    init_table_rows = table['row']

    n_column = len(init_table_header)
    n_row = len(init_table_rows)
    if n_column == 0 or n_row == 0:
        return None

    # [Row] [Col]

    table_list = [init_table_header] + init_table_rows

    for row_idx, row in enumerate(table_list):

        for col_id, value in enumerate(row):
            if value not in linear_nodes:
                head_idx = len(linear_nodes)
                node_dict[value] = head_idx
                linear_nodes.append(value)
            else:
                head_idx = node_dict[value]

            # same column
            for pre_same_col_idx in range(col_id):
                previous_same_col_value = row[pre_same_col_idx]
                assert previous_same_col_value in linear_nodes
                tail_idx = node_dict[previous_same_col_value]

                triples.append((head_idx, '[Col]', tail_idx))

            # same row
            for pre_same_row_idx in range(row_idx):
                previous_same_row_value = table_list[pre_same_row_idx][col_id]
                assert previous_same_row_value in linear_nodes
                tail_idx = node_dict[previous_same_row_value]

                triples.append((head_idx, '[Row]', tail_idx))

    return linear_nodes, triples


def cell_cleanout(cell):
    """

    :param cell: str
    :return:
    """
    cell = cell.replace('\n', ' ')
    return cell


def flat_table(table):
    """

    :param table: dict(id: str, document_title: str, caption: str, document_url: str, alternative_document_urls: str,
    alternative_table_ids: str,
    context_heading: str,
    column: list,
    row: list
    )
    :return:
    """

    flatten_table = []
    init_table_header = table['column']
    init_table_rows = table['row']

    n_column = len(init_table_header)
    n_row = len(init_table_rows)
    if n_column == 0 or n_row == 0:
        return None

    for column_name in init_table_header:
        column_name = cell_cleanout(column_name)
        flatten_table.append(column_name)

    for init_row in init_table_rows:
        for cell in init_row:
            cell = cell_cleanout(cell)
            flatten_table.append(cell)

    return flatten_table


def table_text_overlap_scorer(table, target_sent,
                              cell_text_overlap_score_lower_bound=0.2, lower_match=True):
    """

    :param table:
    :param target_sent:
    :param cell_text_overlap_score_lower_bound:
    :param lower:
    :return:
    """
    n_cell = 0

    n_hit_cell = 0
    for row in table:

        for cell in row:
            text_overlap_score = utils.text_overlap_scorer(hyp=cell, ref=target_sent, lower=lower_match)
            if text_overlap_score > cell_text_overlap_score_lower_bound:
                n_hit_cell += 1
            n_cell += 1

    cell_overlap_score = n_hit_cell / n_cell

    return cell_overlap_score


def table_cleanout(table, target_sent, table_overlap_score_lower_bound=0.2,
                   cell_text_overlap_score_lower_bound=0.2, lower_match=True):
    """

    :param table: dict(id: str, document_title: str, caption: str, document_url: str, alternative_document_urls: str,
    alternative_table_ids: str,
    context_heading: str,
    column: list,
    row: list
    )
    :return:
    """

    table_id = table['id']

    init_table_header = table['column']
    init_table_rows = table['row']

    n_column = len(init_table_header)
    n_row = len(init_table_rows)
    if n_column == 0 or n_row == 0:
        return None

    n_cell = n_column * n_row

    table_header = []
    for column_name in init_table_header:
        column_name = cell_cleanout(column_name)
        table_header.append(column_name)

    table_rows = []
    for init_row in init_table_rows:
        row = []
        for cell in init_row:
            cell = cell_cleanout(cell)
            row.append(cell)
        table_rows.append(row)

    if target_sent is not None:
        cell_overlap_score = table_text_overlap_scorer(table=[table_header] + table_rows,
                                                       target_sent=target_sent,
                                                       cell_text_overlap_score_lower_bound=cell_text_overlap_score_lower_bound,
                                                       lower_match=lower_match)
        if cell_overlap_score < table_overlap_score_lower_bound:
            return None

    new_table = table
    new_table['n_cell'] = n_cell
    new_table['column'] = table_header
    new_table['row'] = table_rows

    return new_table


def text_table_overlap_scorer(value, flatten_table, lower_match=True):
    if lower_match:
        value = value.lower()
    tokens = value.split()
    tokens = [token for token in tokens if token not in stopwords]

    if len(tokens) == 0:
        return 0.0

    if lower_match:
        table_tokens = set(" ".join(flatten_table).lower().split())
    else:
        table_tokens = set(" ".join(flatten_table).split())

    table_tokens = [token for token in list(table_tokens) if token not in stopwords]

    n_token = len(tokens)
    n_hit_token = 0
    for token in tokens:
        if token in table_tokens:
            n_hit_token += 1

    overlap_score = n_hit_token / n_token
    return overlap_score


def extract_special_text_segment(sent):
    res = re.findall(r'\([\s\S]*?\)', sent)
    return res


def target_sent_cleanout(sent, flatten_table, target_segment_overlap_score_lower_bound=0.5, multi_sent=False,
                         lower_match=True):
    """

    :param sent:
    :param flatten_table:
    :param target_segment_overlap_score_lower_bound:
    :param multi_sent:
    :param lower_match:
    :return:
    """

    if multi_sent:
        sent = sent.replace("\n", ' ')
    else:
        sent = sent.split('\n')[0]

    match_text_segments = extract_special_text_segment(sent)
    for text_segment in match_text_segments:
        tmp_text_segment = text_segment[1:-1]
        # print("tmp_text_segment", tmp_text_segment, text_segment)
        # print(sent)
        overlap_score = text_table_overlap_scorer(tmp_text_segment, flatten_table, lower_match=lower_match)
        if overlap_score < target_segment_overlap_score_lower_bound:
            sent = sent.replace(text_segment, '')
    # print("final sent", sent)
    return sent


def questions_cleanout(questions, init_flatten_table, target_segment_overlap_score_lower_bound=0.5, multi_sent=False,
                       lower_match=True):
    """

    :param questions:
    :param init_flatten_table:
    :param target_segment_overlap_score_lower_bound:
    :param multi_sent:
    :param lower_match:
    :return:
    """
    # question_key = ['TITLE', 'DESCRIPTION', 'SEGMENT_TEXT']

    # title = questions['TITLE']
    # description = questions['DESCRIPTION']
    try:
        segment_text = questions['SEGMENT_TEXT']
        target_sent = segment_text
        target_sent = target_sent_cleanout(sent=target_sent, flatten_table=init_flatten_table,
                                           target_segment_overlap_score_lower_bound=target_segment_overlap_score_lower_bound,
                                           multi_sent=multi_sent,
                                           lower_match=lower_match)
    except:
        target_sent = None

    new_question = questions
    new_question['target'] = target_sent

    return new_question


def construct_metadata(table, question):
    metadatas = []

    caption = table['caption']
    context_heading = table['context_heading']
    title = question['TITLE']
    segment_title = question.get('SEGMENT_TITLE', None)
    if len(caption) > 0 or len(context_heading) > 0:
        logger.debug("caption: {}".format(caption))
        logger.debug("context_heading: {}".format(context_heading))
        logger.debug("title: {}".format(title))
        assert False
    if len(title) > 0:
        metadatas.append('The data title is: {}'.format(title))
    if segment_title is not None:
        metadatas.append("The data segment title is: {}".format(segment_title))
    return metadatas


def example_cleanout(data_item,
                     table_overlap_score_lower_bound=0.2,
                     cell_text_overlap_score_lower_bound=0.2,
                     target_segment_overlap_score_lower_bound=0.3,
                     multi_target_sents=False,
                     lower_match=True, only_infobox=True):
    data_id = data_item['id']

    init_questions = data_item['questions']
    init_table = data_item['table']

    table_id = init_table['id']

    data_type = 'infobox'
    flag = table_id.split('/')[-1].split('_')[-1]
    flag = int(flag)
    if flag != 0:
        data_type = 'wikitable'
        if only_infobox:
            return None

    flatten_table = flat_table(init_table)

    # print(data_id)
    # print(init_questions)
    cleanout_questions = questions_cleanout(questions=init_questions, init_flatten_table=flatten_table,
                                            target_segment_overlap_score_lower_bound=target_segment_overlap_score_lower_bound,
                                            multi_sent=multi_target_sents,
                                            lower_match=lower_match)
    # print(data_id, cleanout_questions)
    if cleanout_questions is None:
        return None
    cleanout_target_sent = cleanout_questions['target']
    # print(data_id, cleanout_target_sent)
    if only_infobox:
        if len(cleanout_target_sent) == 0:
            return None

    cleanout_table = table_cleanout(table=init_table, target_sent=cleanout_target_sent,
                                    table_overlap_score_lower_bound=table_overlap_score_lower_bound,
                                    cell_text_overlap_score_lower_bound=cell_text_overlap_score_lower_bound,
                                    lower_match=lower_match)
    if cleanout_table is None:
        return None

    linear_nodes, triples = convert_table_to_graph(cleanout_table)

    metadatas = construct_metadata(cleanout_table, cleanout_questions)
    cleanout_example = dict()
    cleanout_example['id'] = data_id
    cleanout_example['data_type'] = data_type
    cleanout_example['table'] = cleanout_table
    cleanout_example['linear_node'] = linear_nodes
    cleanout_example['triple'] = triples
    cleanout_example['metadata'] = metadatas
    cleanout_example['target_sents'] = [cleanout_target_sent] if cleanout_target_sent is not None else None
    cleanout_example['question'] = cleanout_questions

    return cleanout_example


def cleanout(dataset_file_src, output_src,
             table_overlap_score_lower_bound=0.3, cell_text_overlap_score_lower_bound=0.5,
             target_segment_overlap_score_lower_bound=0.3,
             multi_target_sents=True,
             lower_match=True, only_infobox=True):
    """

    :param dataset_file_src:
    :param output_src:
    :param table_overlap_score_lower_bound:
    :param cell_text_overlap_score_lower_bound:
    :param target_segment_overlap_score_lower_bound:
    :param multi_target_sents:
    :param lower_match:
    :param only_infobox:
    :return:
    """
    data = utils.read_json_file_by_line(dataset_file_src)

    cleanout_examples = []
    n_nodes = 0
    n_enc_tokens = 0
    n_dec_tokens = 0
    n_skip = 0
    n_triples = 0

    max_n_nodes = 0
    max_n_enc_tokens = 0
    max_n_dec_tokens = 0
    max_n_triples = 0
    cnt = 0

    pbar = tqdm(data)
    for data_item in tqdm(data):
        # print("data_item", type(data_item), data_item)
        # print("kg_knowledge", type(kg_knowledge))
        # assert False
        # if cnt == 950:
        #     assert False
        # cnt += 1
        cleanout_example = example_cleanout(data_item=data_item,
                                            table_overlap_score_lower_bound=table_overlap_score_lower_bound,
                                            cell_text_overlap_score_lower_bound=cell_text_overlap_score_lower_bound,
                                            target_segment_overlap_score_lower_bound=target_segment_overlap_score_lower_bound,
                                            multi_target_sents=multi_target_sents,
                                            lower_match=lower_match, only_infobox=only_infobox)
        if cleanout_example is None:
            n_skip += 1
            continue
        n_nodes += len(cleanout_example['linear_node'])
        n_enc_tokens += len(" ".join(cleanout_example['linear_node']).split())
        n_dec_tokens += len(cleanout_example['target_sents'][0].split()) if cleanout_example['target_sents'] is not None else 0
        cleanout_examples.append(cleanout_example)

        pbar.set_description('Skipped {}'.format(n_skip))

    utils.write_to_json_file_by_line(cleanout_examples, output_src)
    n_total_example = len(data)
    n_cleanout_example = len(cleanout_examples)
    logger.info("Finished, total {} examples, cleanout {} examples, "
                "avg_node: {}, avg_enc_tokens: {}, avg_dec_tokens: {}, avg_triples: {}, "
                "cleanout data has been saved at {}".format(n_total_example, n_cleanout_example,
                                                            n_nodes / n_cleanout_example,
                                                            n_enc_tokens / n_cleanout_example,
                                                            n_dec_tokens / n_cleanout_example,
                                                            n_triples / n_cleanout_example,
                                                            output_src))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TaPas Cleanout of argparse')
    parser.add_argument("--tapas_dataset_file_src", type=str, default='../../original_datasets/tapas/tapas.json')
    parser.add_argument("--output_cleanout_file_src", type=str,
                        default='../../cleanout_datasets/tapas/cleanout_tapas_single_sent.json')
    parser.add_argument("--table_overlap_score_lower_bound", type=float, default=0.1)
    parser.add_argument("--cell_text_overlap_score_lower_bound", type=float, default=0.3)
    parser.add_argument("--target_segment_overlap_score_lower_bound", type=float, default=0.3)
    parser.add_argument("--multi_target_sents", type=ast.literal_eval, default=True)
    parser.add_argument("--lower_match", type=ast.literal_eval, default=True)
    parser.add_argument("--only_infobox", type=ast.literal_eval, default=True)
    args = parser.parse_args()
    print_args(args)

    cleanout(dataset_file_src=args.tapas_dataset_file_src, output_src=args.output_cleanout_file_src,
             table_overlap_score_lower_bound=args.table_overlap_score_lower_bound,
             cell_text_overlap_score_lower_bound=args.cell_text_overlap_score_lower_bound,
             target_segment_overlap_score_lower_bound=args.target_segment_overlap_score_lower_bound,
             multi_target_sents=args.multi_target_sents,
             lower_match=args.lower_match,
             only_infobox=args.only_infobox)
