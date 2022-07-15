""" Contains code for encoding an input sequence. """

import torch
import torch.nn.functional as F
from .torch_utils import create_multilayer_lstm_params, encode_sequence

class Encoder(torch.nn.Module):
    """ Encodes an input sequence. """
    def __init__(self, num_layers, input_size, state_size):
        super().__init__()

        self.num_layers = num_layers
        self.forward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-ef")
        self.backward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-eb")

    def forward(self, sequence, embedder, dropout_amount=0.):
        """ Encodes a sequence forward and backward.
        Inputs:
            forward_seq (list of str): The string forwards.
            backward_seq (list of str): The string backwards.
            f_rnns (list of dy.RNNBuilder): The forward RNNs.
            b_rnns (list of dy.RNNBuilder): The backward RNNS.
            emb_fn (dict str->dy.Expression): Embedding function for tokens in the
                sequence.
            size (int): The size of the RNNs.
            dropout_amount (float, optional): The amount of dropout to apply.

        Returns:
            (list of dy.Expression, list of dy.Expression), list of dy.Expression,
            where the first pair is the (final cell memories, final cell states) of
            all layers, and the second list is a list of the final layer's cell
            state for all tokens in the sequence.
        """
        forward_state, forward_outputs = encode_sequence(
            sequence,
            self.forward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        backward_state, backward_outputs = encode_sequence(
            sequence[::-1],
            self.backward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        cell_memories = []
        hidden_states = []
        for i in range(self.num_layers):
            cell_memories.append(torch.cat([forward_state[0][i], backward_state[0][i]], dim=0))
            hidden_states.append(torch.cat([forward_state[1][i], backward_state[1][i]], dim=0))

        assert len(forward_outputs) == len(backward_outputs)

        backward_outputs = backward_outputs[::-1]

        final_outputs = []
        for i in range(len(sequence)):
            final_outputs.append(torch.cat([forward_outputs[i], backward_outputs[i]], dim=0))

        return (cell_memories, hidden_states), final_outputs