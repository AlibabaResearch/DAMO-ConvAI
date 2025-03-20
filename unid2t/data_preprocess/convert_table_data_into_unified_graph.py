import sys

import constants


def convert_table_to_triples(table_header, rows):

    linearized_table = []
    triples = set()

    node_dict = dict()

    for i, header in enumerate(table_header):

        if header in node_dict:
            header_idx = node_dict[header]
        else:
            header_idx = len(node_dict)
            node_dict[header] = header_idx
            linearized_table.append(header)

        if i > 0:
            for j in range(0, i - 1):
                sub_header = table_header[j]
                sub_header_idx = node_dict[sub_header]
                triples.add((header_idx, sub_header_idx))

    for row_idx, row in enumerate(rows):

        # same row
        for col_idx, cell in enumerate(row):

            corres_header = table_header[col_idx]
            header_idx = node_dict[corres_header]

            if cell in node_dict:
                cell_idx = node_dict[cell]
            else:
                cell_idx = len(node_dict)
                node_dict[cell_idx] = cell
                linearized_table.append(cell)

            triples.add((header_idx, cell_idx))

            # same row
            if col_idx > 0:
                for previous_same_row_col_idx in range(0, col_idx - 1):
                    previous_same_row_col_cell = row[previous_same_row_col_idx]
                    previous_same_row_col_cell_idx = node_dict[previous_same_row_col_cell]
                    triples.add((cell_idx, previous_same_row_col_cell_idx))

            # same col
            if row_idx > 0:
                for precious_same_col_row_idx in range(0, row_idx - 1):
                    precious_same_col_row_cell = rows[precious_same_col_row_idx][col_idx]
                    precious_same_col_row_cell_idx = node_dict[precious_same_col_row_cell]
                    triples.add((cell_idx, precious_same_col_row_cell_idx))

    triples = list(triples)
    return linearized_table, triples









