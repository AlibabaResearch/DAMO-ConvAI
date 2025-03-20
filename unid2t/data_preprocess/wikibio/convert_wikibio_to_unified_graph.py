import sys
import os
from copy import deepcopy
import numpy as np

from tqdm import tqdm
import argparse

from data_preprocess import utils


def get_heuristic_max_row_and_column_number(totto_table):
    max_row_number = 0
    column_number = 0
    for i, row in enumerate(totto_table):
        max_column_span = 0
        max_row_span = 0

        cur_column_number = 0
        for j, cell in enumerate(row):
            column_span = cell['column_span']
            # is_header = cell['is_header']
            row_span = cell['row_span']
            if column_span > max_column_span:
                max_column_span = column_span

            if row_span > max_row_span:
                max_row_span = row_span

            cur_column_number += column_span

        if i == 0:
            column_number = cur_column_number

        # assert column_number == cur_column_number


        max_row_number += max_row_span
        # max_column_number += max_column_span

    return max_row_number, column_number


def heuristic_search_cell_fill_position(heuristic_flags):
    """

    :param heuristic_flags: np.arrary(max_n_row, max_n_column)
    :return:
    """
    max_n_row = heuristic_flags.shape[0]
    max_n_column = heuristic_flags.shape[1]
    for i in range(max_n_row):
        for j in range(max_n_column):
            if heuristic_flags[i][j] == 0:
                return i, j

    return None, None


def construct_heuristic_table_from_totto_table(totto_table):
    """

    :param totto_table: list(
                row: list(
                    column: dict(
                            column_span: int,
                            is_header: bool
                            row_span: int,
                            value: str
                    )
                )
        )
    :return:
    """
    max_row_number, max_column_number = get_heuristic_max_row_and_column_number(totto_table)
    heuristic_flags = np.zeros([max_row_number, max_column_number])
    # heuristic_table = [(-1, -1, -1) for i in range(max_column_number)] * max_row_number
    heuristic_table = [[(-1, -1, -1) for i in range(max_column_number)] for j in range(max_row_number)]
    heuristic_table = np.asarray(heuristic_table)
    # print("heuristic_table", max_column_number, max_row_number, heuristic_table)
    flatten_cells = []

    adjusted_totto_table = []
    for i, row in enumerate(totto_table):
        adjusted_totto_row = []
        for j, cell in enumerate(row):
            column_span = cell['column_span']
            # is_header = cell['is_header']
            row_span = cell['row_span']
            # print(i, j, heuristic_flags)
            cur_start_row_pos, cur_start_col_pos = heuristic_search_cell_fill_position(heuristic_flags)
            # print("searched row_start_pos and col_start_pos are: {}, {}".format(cur_start_row_pos, cur_start_col_pos))
            assert cur_start_row_pos is not None, cell['value']

            heuristic_flags[cur_start_row_pos:cur_start_row_pos + row_span,
            cur_start_col_pos:cur_start_col_pos + column_span] = 1

            cell_idx = len(flatten_cells)
            heuristic_table[cur_start_row_pos:cur_start_row_pos + row_span,
            cur_start_col_pos:cur_start_col_pos + column_span] = (cell_idx, i, j)

            adjusted_cell = deepcopy(cell)
            adjusted_cell['node_id'] = cell_idx
            flatten_cells.append(adjusted_cell)

            adjusted_totto_row.append(adjusted_cell)

        adjusted_totto_table.append(adjusted_totto_row)
    # print("heuristic_table", heuristic_table)
    return heuristic_flags, heuristic_table, adjusted_totto_table, flatten_cells


def obtain_abs_coordinates_for_cells(heuristic_flags, heuristic_table, flatten_cells):
    """

    :param heuristic_flags:
    :param heuristic_table:
    :param flatten_cells:
    :return:
    """
    abs_coordinates_for_flatten_cells = []
    max_h = heuristic_flags.shape[0]
    max_w = heuristic_flags.shape[1]

    for cell in flatten_cells:
        target_node_id = cell['node_id']
        abs_coordinates = []
        for x_cor in range(max_h):
            for y_cor in range(max_w):
                if heuristic_flags[x_cor][y_cor] == 1:
                    tmp_node_id, _, _ = heuristic_table[x_cor][y_cor]
                    if tmp_node_id == target_node_id:
                        abs_coordinates.append((x_cor, y_cor))

        assert len(abs_coordinates) > 0
        abs_coordinates_for_flatten_cells.append(abs_coordinates)

    return abs_coordinates_for_flatten_cells


def convert_highlight_subtable_into_triples(highlighted_cell_indices, adjusted_totto_table, abs_coordinates_for_flatten_cells):
    """

    :param highlighted_cell_indices:
    :param adjusted_totto_table:
    :param abs_coordinates_for_flatten_cells:
    :return:
    """
    highlighted_cells = []
    for row in adjusted_totto_table:
        for cell in row:
            is_header = cell['is_header']
            if is_header:
                cell_value = cell['value']
                if len(cell_value):
                    highlighted_cells.append(cell)

    for highlighted_cell_index in highlighted_cell_indices:
        x_cor, y_cor = highlighted_cell_index[0], highlighted_cell_index[1]
        highlighted_cell = adjusted_totto_table[x_cor][y_cor]
        highlighted_cells.append(highlighted_cell)

    table_header = []
    table_values = []

    # Retag
    for highlighted_cell in highlighted_cells:
        highlighted_node_id = len(table_values)
        highlighted_cell['highlighted_node_id'] = highlighted_node_id
        is_header = highlighted_cell['is_header']
        if is_header:
            table_header.append(highlighted_cell['value'])
        table_values.append(highlighted_cell['value'])

    triples = []
    for highlighted_cell in highlighted_cells:
        node_id = highlighted_cell['node_id']
        head_id = highlighted_cell['highlighted_node_id']
        head_abs_coordinates = abs_coordinates_for_flatten_cells[node_id]
        for head_abs_coordinate in head_abs_coordinates:
            head_x_cor, head_y_cor = head_abs_coordinate
            for tail_highlighted_cell in highlighted_cells:
                tail_node_id = tail_highlighted_cell['node_id']
                if node_id == tail_node_id:
                    continue
                tail_id = tail_highlighted_cell['highlighted_node_id']
                tail_abs_coordinates = abs_coordinates_for_flatten_cells[tail_node_id]

                for tail_abs_coordinate in tail_abs_coordinates:
                    tail_x_cor, tail_y_cor = tail_abs_coordinate
                    if head_x_cor == tail_x_cor:
                        triples.append((head_id, '[Row]', tail_id))

                    if head_y_cor == tail_y_cor:
                        triples.append((head_id, '[Col]', tail_id))

    return table_header, table_values, triples


def construct_meta_data(table_page_title, table_section_title, table_section_text):
    meta_datas = []
    if len(table_page_title):
        meta_sent = "The table page title is: " + table_page_title
        if meta_sent[-1] != '.':
            meta_sent = meta_sent + '.'
        meta_datas.append(meta_sent)

    if len(table_section_title):
        meta_sent = "The table section title is: " + table_section_title
        if meta_sent[-1] != '.':
            meta_sent = meta_sent + '.'
        meta_datas.append(meta_sent)

    if len(table_section_text):
        meta_sent = "The table page text is: " + table_section_text
        if meta_sent[-1] != '.':
            meta_sent = meta_sent + '.'
        meta_datas.append(meta_sent)

    return meta_datas


def convert_wikibio_example_to_unified_graph(example):
    """

    :param example: dict(
        table_page_title: str,
        table_webpage_url: str,
        table_section_title: str,
        table_section_text: str,
        highlighted_cells: list(pair())
        example_id:
        sentence_annotations: dict(
            original_sentence: str,
            sentence_after_deletion: str,
            sentence_after_ambiguity: str,
            final_sentence: str
        )
        table: list(
                row: list(
                    column: dict(
                            column_span: int,
                            is_header: bool
                            row_span: int,
                            value: str
                    )
                )
        )

    :return:
    """

    linear_node = example['linear_node']
    target_sents = example['target_sents']
    linear_node_new = []
    triples = []
    k_list = []
    v_list = []
    converted_example = dict()

    for kv in linear_node:
        k,v = kv.split(":")
        if k == "article_title":
            meta_data = ["the article title is: "+kv.split(":")[1]]
            continue
        if k not in k_list:
            k_list.append(k)
        if v not in v_list:
            v_list.append(v)
        linear_node_new.append(k)
        linear_node_new.append(v)
        triples.append([linear_node_new.index(k),linear_node_new.index(v)])

    for i in range(len(k_list)):
        for j in range(i + 1 ,len(k_list)):
            triples.append([linear_node_new.index(k_list[i]) , linear_node_new.index(k_list[j])])
    for i in range(len(v_list)):
        for j in range(i + 1 ,len(v_list)):
            triples.append([linear_node_new.index(v_list[i]) , linear_node_new.index(v_list[j])])



    converted_example['linear_node'] = linear_node_new
    converted_example['triple'] = triples

    if target_sents is not None:
        converted_example['target_sents'] = target_sents
    converted_example['metadata'] = meta_data

    return converted_example


def convert(totto_file, output_file):
    examples = utils.read_json_file_by_line(totto_file)

    converted_examples = []
    bar = tqdm(examples)
    i = 0
    for example in bar:
        # bar.set_description('{}-th'.format(i))
        # if i < 6408:
        #     i += 1
        #     continue
        converted_example = convert_wikibio_example_to_unified_graph(example)
        converted_examples.append(converted_example)
        i += 1

    utils.write_to_json_file_by_line(converted_examples, output_file)
    print("Finished, converted {} examples in {}, the converted data been saved at: {}".format(len(converted_examples),
                                                                                               totto_file,
                                                                                               output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert totto to unified graph')
    parser.add_argument("--original_totto_dir", type=str, default='../../orig_datasets/text_wikibio')
    parser.add_argument("--output_dir", type=str,
                        default='../../cleanout_datasets/wikibio_meta')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    original_totto_files = os.listdir(args.original_totto_dir)
    for original_totto_file in original_totto_files:
        print("original_totto_file", original_totto_file)
        if not original_totto_file.endswith('.json'):
            continue
        # if 'train' in original_totto_file or 'dev' in original_totto_file:
        #     continue
        original_totto_file_src = os.path.join(args.original_totto_dir, original_totto_file)
        print(original_totto_file_src)
        converted_file_src = os.path.join(args.output_dir, original_totto_file)
        convert(original_totto_file_src, converted_file_src)




