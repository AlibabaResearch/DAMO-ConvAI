#coding=utf8
""" ONLSTM and traditional LSTM with locked dropout """
import torch
import torch.nn as nn
import torch.nn.functional as F

def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)

class LinearDropConnect(nn.Linear):
    """ Used in recurrent connection dropout """
    def __init__(self, in_features, out_features, bias=True, dropconnect=0.):
        super(LinearDropConnect, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.dropconnect = dropconnect

    def sample_mask(self):
        if self.dropconnect == 0.:
            self._weight = self.weight.clone()
        else:
            mask = self.weight.new_zeros(self.weight.size(), dtype=torch.bool)
            mask.bernoulli_(self.dropconnect)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, inputs, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(inputs, self._weight, self.bias) # apply the same mask to weight matrix in linear module
        else:
            return F.linear(inputs, self.weight * (1 - self.dropconnect), self.bias)

class LockedDropout(nn.Module):
    """ Used in dropout between layers """
    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(LockedDropout, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def sample_masks(self, x):
        self.masks = []
        for _ in range(self.num_layers - 1):
            mask = x.new_zeros(x.size(0), 1, self.hidden_size).bernoulli_(1 - self.dropout)
            mask = mask.div_(1 - self.dropout)
            mask.requires_grad = False
            self.masks.append(mask)

    def forward(self, x, layer=0):
        """ x: bsize x seqlen x hidden_size """
        if (not self.training) or self.dropout == 0. or layer == self.num_layers - 1: # output hidden states, no dropout
            return x
        mask = self.masks[layer]
        mask = mask.expand_as(x)
        return mask * x

class RecurrentNeuralNetwork(nn.Module):
    def init_hiddens(self, x):
        return x.new_zeros(self.num_layers, x.size(0), self.hidden_size), \
            x.new_zeros(self.num_layers, x.size(0), self.hidden_size)

    def forward(self, inputs, hiddens=None, start=False, layerwise=False):
        """
        @args:
            start: whether sampling locked masks for recurrent connections and between layers
            layerwise: whether return a list, results of intermediate layer outputs
        @return:
            outputs: bsize x seqlen x hidden_size
            final_hiddens: hT and cT, each of size: num_layers x bsize x hidden_size
        """
        assert inputs.dim() == 3
        if hiddens is None:
            hiddens = self.init_hiddens(inputs)
        bsize, seqlen, _ = list(inputs.size())
        prev_state = list(hiddens) # each of size: num_layers, bsize, hidden_size
        prev_layer = inputs # size: bsize, seqlen, input_size
        each_layer_outputs, final_h, final_c = [], [], []

        if self.training and start:
            for c in self.cells:
                c.sample_masks()
            self.locked_dropout.sample_masks(inputs)

        for l in range(len(self.cells)):
            curr_layer = [None] * seqlen
            curr_inputs = self.cells[l].ih(prev_layer)
            next_h, next_c = prev_state[0][l], prev_state[1][l]
            for t in range(seqlen):
                hidden, cell = self.cells[l](None, (next_h, next_c), transformed_inputs=curr_inputs[:, t])
                next_h, next_c = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden

            prev_layer = torch.stack(curr_layer, dim=1) # bsize x seqlen x hidden_size
            each_layer_outputs.append(prev_layer)
            final_h.append(next_h)
            final_c.append(next_c)
            prev_layer = self.locked_dropout(prev_layer, layer=l)

        outputs, final_hiddens = prev_layer, (torch.stack(final_h, dim=0), torch.stack(final_c, dim=0))
        if layerwise:
            return outputs, final_hiddens, each_layer_outputs
        else:
            return outputs, final_hiddens

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, dropconnect=0.):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hh = LinearDropConnect(hidden_size, hidden_size * 4, bias=bias, dropconnect=dropconnect)
        self.drop_weight_modules = [self.hh]

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()

    def forward(self, inputs, hiddens, transformed_inputs=None):
        """
            inputs: bsize x input_size
            hiddens: tuple of h0 (bsize x hidden_size) and c0 (bsize x hidden_size)
            transformed_inputs: short cut for inputs, save time if seq len is already provied in training
            return tuple of h1 (bsize x hidden_size) and c1 (bsize x hidden_size)
        """
        if transformed_inputs is None:
            transformed_inputs = self.ih(inputs)
        h0, c0 = hiddens
        gates = transformed_inputs + self.hh(h0)
        ingate, forgetgate, outgate, cell = gates.contiguous().\
            view(-1, 4, self.hidden_size).chunk(4, 1)
        forgetgate = torch.sigmoid(forgetgate.squeeze(1))
        ingate = torch.sigmoid(ingate.squeeze(1))
        cell = torch.tanh(cell.squeeze(1))
        outgate = torch.sigmoid(outgate.squeeze(1))
        c1 = forgetgate * c0 + ingate * cell
        h1 = outgate * torch.tanh(c1)
        return h1, c1

class LSTM(RecurrentNeuralNetwork):

    def __init__(self, input_size, hidden_size, num_layers=1, chunk_num=1, bias=True, dropout=0., dropconnect=0.):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [LSTMCell(input_size, hidden_size, bias, dropconnect)] +
            [LSTMCell(hidden_size, hidden_size, bias, dropconnect) for i in range(num_layers - 1)]
        )
        self.locked_dropout = LockedDropout(hidden_size, num_layers, dropout) # dropout rate between layers

class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_num=8, bias=True, dropconnect=0.2):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_num = chunk_num # chunk_num should be divided by hidden_size
        if self.hidden_size % self.chunk_num != 0:
            raise ValueError('[Error]: chunk number must be divided by hidden size in ONLSTM Cell')
        self.chunk_size = int(hidden_size / chunk_num)

        self.ih = nn.Linear(input_size, self.chunk_size * 2 + hidden_size * 4, bias=bias)
        self.hh = LinearDropConnect(hidden_size, self.chunk_size * 2 + hidden_size * 4, bias=bias, dropconnect=dropconnect)
        self.drop_weight_modules = [self.hh]

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()

    def forward(self, inputs, hiddens, transformed_inputs=None):
        """
            inputs: bsize x input_size
            hiddens: tuple of h0 (bsize x hidden_size) and c0 (bsize x hidden_size)
            transformed_inputs: short cut for inputs, save time if seq len is already provied in training
            return tuple of h1 (bsize x hidden_size) and c1 (bsize x hidden_size)
        """
        if transformed_inputs is None:
            transformed_inputs = self.ih(inputs)
        h0, c0 = hiddens
        gates = transformed_inputs + self.hh(h0)
        cingate, cforgetgate = gates[:, :self.chunk_size * 2].chunk(2, 1)
        ingate, forgetgate, outgate, cell = gates[:, self.chunk_size * 2:].contiguous().\
            view(-1, self.chunk_size * 4, self.chunk_num).chunk(4, 1)

        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)
        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        forgetgate = torch.sigmoid(forgetgate)
        ingate = torch.sigmoid(ingate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        c0 = c0.contiguous().view(-1, self.chunk_size, self.chunk_num)
        c1 = forgetgate * c0 + ingate * cell
        h1 = outgate * torch.tanh(c1)
        return h1.contiguous().view(-1, self.hidden_size), c1.contiguous().view(-1, self.hidden_size)

class ONLSTM(RecurrentNeuralNetwork):

    def __init__(self, input_size, hidden_size, num_layers=1, chunk_num=8, bias=True, dropout=0., dropconnect=0.):
        super(ONLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [ONLSTMCell(input_size, hidden_size, chunk_num, bias, dropconnect)] +
            [ONLSTMCell(hidden_size, hidden_size, chunk_num, bias, dropconnect) for i in range(num_layers - 1)]
        )
        self.locked_dropout = LockedDropout(hidden_size, num_layers, dropout) # dropout rate between layers
