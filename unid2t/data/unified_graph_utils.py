import numpy as np
import torch
from transformers import PreTrainedTokenizer
import constants


def convert_data_to_unified_graph(linearized_nodes, triples, tokenizer: PreTrainedTokenizer, special_tokens,
                                  metadatas=None,
                                  prefix=None,
                                  lower=False,
                                  segment_token_full_connection=True,
                                  position_style = "inf"):
    """

    :param linearized_nodes: list
    :param triples: list(head_idx, tail_idx)
    :param tokenizer:
    :param special_tokens: list
    :param metadatas: list()
    :param prefix:
    :param lower
    :param segment_token_full_connection
    :return:
    """

    tokens = []

    if prefix is not None and len(prefix) > 0:
        if lower:
            prefix = prefix.lower()
        prefix_tokens = tokenizer.tokenize(prefix)
        tokens.extend(prefix_tokens)

    metadata_flag_poses = []
    if metadatas is not None:
        metadata_flag_token = constants.METADATA_TOKEN
        for metadata in metadatas:
            if lower:
                metadata = metadata.lower()
            metadata_flag_poses.append(len(tokens))
            metadata_with_flag = metadata_flag_token + " " + metadata
            metadata_tokens = tokenizer.tokenize(metadata_with_flag)
            tokens.extend(metadata_tokens)

    linearized_nodes = [constants.NODE_TOKEN + ' ' + node for node in linearized_nodes]

    graph_token_start_pos = len(tokens)
    linearized_node_tokens, graph_connection_matrix, \
    linearized_node_token_segments, \
    linear_table_relative_position_matrix = tokenized_graph_data(linearized_nodes=linearized_nodes,
                                                                 triples=triples,
                                                                 tokenizer=tokenizer,
                                                                 special_tokens=special_tokens,
                                                                 lower=lower,
                                                                 position_style= position_style)
    tokens.extend(linearized_node_tokens)
    n_token = len(tokens)
    # print("n_token" , n_token)
    # construct for connection_matrix
    connection_matrix = np.zeros([n_token, n_token])

    # prefix, metadata tokens are connected to all graph node tokens
    connection_matrix[:graph_token_start_pos] = 1
    # copy graph connection matrix
    connection_matrix[graph_token_start_pos:, graph_token_start_pos:] = graph_connection_matrix

    # construct for relative bias
    relative_position_matrix = np.arange(n_token)[None, :] - np.arange(n_token)[:, None]

    if segment_token_full_connection:
        segment_token_relative_positions = []
        for linearized_node_token_segment in linearized_node_token_segments:
            linearized_node_token_start_pos, linearized_node_token_end_pos = linearized_node_token_segment
            global_linearized_node_token_start_pos = linearized_node_token_start_pos + graph_token_start_pos
            connection_matrix[global_linearized_node_token_start_pos] = 1

            segment_token_relative_positions.append((global_linearized_node_token_start_pos,
                                                     relative_position_matrix[global_linearized_node_token_start_pos]))

        relative_position_matrix[graph_token_start_pos:, graph_token_start_pos:] = linear_table_relative_position_matrix

        """
        waiting to be optimized
        """
        for global_segment_token_pos, segment_token_relative_position in segment_token_relative_positions:
            relative_position_matrix[global_segment_token_pos] = segment_token_relative_position

    else:
        relative_position_matrix[graph_token_start_pos:, graph_token_start_pos:] = linear_table_relative_position_matrix

    assert connection_matrix.shape == relative_position_matrix.shape
    return tokens, connection_matrix, relative_position_matrix


def construct_relative_position(connection_matrix):
    """

    :param connection_matrix: numpy(n_token, n_token), connection_matrix[i][j] denotes the i-th node should attend
           to j-th node
    :return:
    """
    n_token = connection_matrix.shape[0]
    relative_position_matrix = np.zeros([n_token, n_token])
    for i_token_pos in range(n_token):
        centre_token_pos = 0
        if i_token_pos > 0:
            for j_token_pos in range(0, i_token_pos):
                if connection_matrix[i_token_pos][j_token_pos] == 1:
                    centre_token_pos += 1
        # print("centre_token_pos", centre_token_pos)
        surround_token_pos = 0
        for j_token_pos in range(n_token):
            flag = connection_matrix[i_token_pos][j_token_pos]
            if flag == 1:
                relative_position = surround_token_pos - centre_token_pos
                relative_position_matrix[i_token_pos][j_token_pos] = relative_position

                surround_token_pos += 1

    return relative_position_matrix


def tokenized_graph_data(linearized_nodes, triples, tokenizer: PreTrainedTokenizer, special_tokens,
                         lower=False, initial_relative_position=0, position_style = "inf"):
    """

    :param linearized_nodes:
    :param triples:
    :param tokenizer
    :param lower
    :param initial_relative_position
    :return:
    """

    linearized_node_tokens = []
    node_token_segments = []  # [start_pos, end_pos)
    n_total = 0
    for node in linearized_nodes:
        if lower:
            node = node.lower()
        node_tokens = tokenizer.tokenize(node)

        node_token_start_pos = n_total
        node_token_end_pos = n_total + len(node_tokens)

        node_token_segments.append((node_token_start_pos, node_token_end_pos))

        linearized_node_tokens.extend(node_tokens)
        n_total = node_token_end_pos

    connection_matrix = np.eye(n_total, n_total)
    linear_table_relative_position_matrix = np.zeros([n_total, n_total])


    if "inf" in position_style:
        for i in range(n_total):
            for j in range(n_total):
                if j > i:
                    linear_table_relative_position_matrix[i][j] = np.inf
                elif j < i:
                    linear_table_relative_position_matrix[i][j] = -np.inf
    else:
        linear_table_relative_position_matrix.fill(initial_relative_position)

    for triple in triples:
        head_node_idx = triple[0]
        tail_node_idx = triple[-1]
        if tail_node_idx < head_node_idx:
            tmp = head_node_idx
            head_node_idx = tail_node_idx
            tail_node_idx = tmp

        # node inter tokens self-connections
        head_node_token_start_pos, head_node_token_end_pos = node_token_segments[head_node_idx]

        # n_head_node_token = head_node_token_end_pos - head_node_token_start_pos
        # head_node_token_full_connection_matrix = np.ones(n_head_node_token, n_head_node_token)

        # connection_matrix[head_node_token_start_pos:head_node_token_end_pos,
        # head_node_token_start_pos:head_node_token_end_pos] = head_node_token_full_connection_matrix
        connection_matrix[head_node_token_start_pos:head_node_token_end_pos,
        head_node_token_start_pos:head_node_token_end_pos] = 1

        tail_node_token_start_pos, tail_node_token_end_pos = node_token_segments[tail_node_idx]

        # n_tail_node_token = tail_node_token_end_pos - tail_node_token_start_pos
        # tail_node_token_full_connection_matrix = np.ones(n_tail_node_token, n_tail_node_token)

        # connection_matrix[tail_node_token_start_pos:tail_node_token_end_pos,
        # tail_node_token_start_pos:tail_node_token_end_pos] = tail_node_token_full_connection_matrix
        connection_matrix[tail_node_token_start_pos:tail_node_token_end_pos,
        tail_node_token_start_pos:tail_node_token_end_pos] = 1

        # forward connection
        connection_matrix[head_node_token_start_pos:head_node_token_end_pos,
        tail_node_token_start_pos:tail_node_token_end_pos] = 1

        # reverse connection
        connection_matrix[tail_node_token_start_pos:tail_node_token_end_pos,
        head_node_token_start_pos:head_node_token_end_pos] = 1

        # relative position
        n_head_node_token = head_node_token_end_pos - head_node_token_start_pos
        n_tail_node_token = tail_node_token_end_pos - tail_node_token_start_pos
        # (n_head_node_token + n_tail_node_token) * (n_head_node_token + n_tail_node_token)
        two_segment_relative_position = construct_relative_position_for_two_segment(n_head_node_token,
                                                                                    n_tail_node_token)

        # head-head
        linear_table_relative_position_matrix[head_node_token_start_pos:head_node_token_end_pos,
        head_node_token_start_pos:head_node_token_end_pos] = two_segment_relative_position[:n_head_node_token,
                                                             :n_head_node_token]
        # head-tail
        linear_table_relative_position_matrix[head_node_token_start_pos:head_node_token_end_pos,
        tail_node_token_start_pos:tail_node_token_end_pos] = two_segment_relative_position[:n_head_node_token,
                                                             n_head_node_token:]

        # tail-tail
        linear_table_relative_position_matrix[tail_node_token_start_pos:tail_node_token_end_pos,
        tail_node_token_start_pos:tail_node_token_end_pos] = two_segment_relative_position[n_head_node_token:,
                                                             n_head_node_token:]

        # tail-head
        linear_table_relative_position_matrix[tail_node_token_start_pos:tail_node_token_end_pos,
        head_node_token_start_pos:head_node_token_end_pos] = two_segment_relative_position[n_head_node_token:,
                                                             :n_head_node_token]

    assert connection_matrix.shape == linear_table_relative_position_matrix.shape

    return linearized_node_tokens, connection_matrix, node_token_segments, linear_table_relative_position_matrix


def construct_relative_position_for_two_segment(n_head_node_token, n_tail_node_token):
    context_position = np.arange(n_head_node_token + n_tail_node_token)[:, None]
    head_position = np.arange(n_head_node_token + n_tail_node_token)[None, :]

    relative_position = head_position - context_position
    return relative_position
