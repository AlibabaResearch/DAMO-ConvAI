import torch
from torch import Tensor
from typing import List
import numpy as np


def split_matrix(matrix: Tensor, lengths: List, reduction='mean') -> Tensor:
    """
    :param matrix:  torch.tensor
    :param lengths: list
    :return:
    """
    output_matrix = torch.zeros(size=(len(lengths), len(lengths)))
    cumsum_lengths = np.cumsum([0] + lengths)
    for i in range(len(lengths)):
        for j in range(len(lengths)):
            splited_matrix_block = matrix[cumsum_lengths[i]:cumsum_lengths[i+1], cumsum_lengths[j]:cumsum_lengths[j+1]]
            if reduction == 'mean':
                output_matrix[i, j] = splited_matrix_block.mean()
            elif reduction == 'sum':
                output_matrix[i, j] = splited_matrix_block.sum()
            elif reduction == 'max':
                output_matrix[i, j] = splited_matrix_block.max()
            elif reduction == 'min':
                output_matrix[i, j] = splited_matrix_block.min()
            else:
                raise ValueError('reduction=[%s] has not been supported.' % reduction)
    return output_matrix


if __name__ == '__main__':
    matrix = torch.randint(0, 3, size=(10, 10), dtype=torch.float)
    lengths = [2, 3, 4, 1]
    output_matrix = split_matrix(matrix, lengths, 'min')

    print(matrix)
    print(output_matrix)
