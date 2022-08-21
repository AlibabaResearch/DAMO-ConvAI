import torch.nn as nn
import numpy as np
import torch
from .Layer import *
from sqlova.utils.utils_wikisql import *
def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

class ParserModel(nn.Module):
    def __init__(self, input_dims, lstm_hiddens, lstm_layers, dropout_lstm_input, \
                  dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, training, dropout_mlp, n_dep_ops):
        super(ParserModel, self).__init__()
        self.input_dims = input_dims
        self.lstm_hiddens = lstm_hiddens
        self.lstm_layers = lstm_layers
        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        self.training = training
        self.dropout_mlp = dropout_mlp
        self.n_dep_ops = n_dep_ops

        self.enc_h = nn.LSTM(input_size=self.input_dims, hidden_size=int(self.lstm_hiddens / 2),
                             num_layers=self.lstm_layers, batch_first=True,
                             dropout=self.dropout_lstm_hidden, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=self.input_dims, hidden_size=int(self.lstm_hiddens / 2),
                             num_layers=self.lstm_layers, batch_first=True,
                             dropout=self.dropout_lstm_hidden, bidirectional=True)

        # self.lstm = MyLSTM(
        #     input_size = self.input_dims,
        #     hidden_size = self.lstm_hiddens,
        #     num_layers = self.lstm_layers,
        #     batch_first = True,
        #     bidirectional = True,
        #     dropout_in = self.dropout_lstm_input,
        #     dropout_out = self.dropout_lstm_hidden,
        # )
        self.log_vars = nn.Parameter(torch.zeros((2)))

        self.mlp_arc_dep = NonLinear(
            input_size = self.lstm_hiddens,
            hidden_size = self.mlp_arc_size + self.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size = self.lstm_hiddens,
            hidden_size = self.mlp_arc_size + self.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))

        self.total_num = int((self.mlp_arc_size+self.mlp_rel_size) / 100)
        self.arc_num = int(self.mlp_arc_size / 100)
        self.rel_num = int(self.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(self.mlp_arc_size, self.mlp_arc_size, \
                                     1, bias=(True, False))
        self.rel_biaffine = Biaffine(self.mlp_rel_size, self.mlp_rel_size, \
                                     self.n_dep_ops, bias=(True, True))

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs):
    # def forward(self, output_pre, masks):
        # mask_bert = torch.unsqueeze(masks, dim=2)
        # mask_bert = mask_bert.expand(-1, -1, self.input_dims)
        # pre_bert = output_pre * mask_bert
        # # print("output_pre: ", output_pre.shape) 
        # # print("masks: ", masks)       

        # if self.training:
        #     pre_bert = drop_sequence_sharedmask(pre_bert, self.dropout_mlp)

        # outputs, _ = self.lstm(pre_bert, masks, None)
        # outputs = outputs.transpose(1, 0)

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)

        bS = len(l_hs)
        f_n = [l_n[i] + l_hs[i] for i in range(bS)]
        mL_n = max(f_n)
        outputs = torch.zeros([bS, mL_n, self.lstm_hiddens]).to(device)
        for b, ln in enumerate(l_n):
            outputs[b, 0:ln, :] = wenc_n[b, 0:ln, :]
            outputs[b, ln:ln+l_hs[b], :] = wenc_hs[b, 0:l_hs[b], :]

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, split_size_or_sections=100, dim=2)
        x_all_head_splits = torch.split(x_all_head, split_size_or_sections=100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)

        return arc_logit, rel_logit_cond, f_n


class BiaffineParser(object):
    def __init__(self, model):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs):
        # batch_size = len(l_n_amr)
        # length = max(l_n_amr)
        # masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
        # for i, ln in enumerate(l_n_amr):
        #     for j in range(ln):
        #         masks[i, j] = 1
        # if self.use_cuda:
        #     masks = masks.cuda(self.device)

        arc_logits, rel_logits, f_n = self.model.forward(wemb_n, l_n, wemb_hpu, l_hpu, l_hs)
        self.arc_logits = arc_logits
        self.rel_logits = rel_logits
        self.f_n = f_n

        return arc_logits, rel_logits, f_n

    def compute_loss(self, true_arcs, true_rels, lengths, part_masks):
        if self.use_cuda:
            part_masks = part_masks.cuda(self.device)

        b, l1, l2 = self.arc_logits.size()
        #print("true_arcs: ", true_arcs)
        #print("true_rels: ", true_rels)
        #print("arc_logits.size: ", self.arc_logits.size())
        #print("rel_logits.size: ", self.rel_logits.size())
        index_true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64))
        true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64))

        # print('idx true arcs:', index_true_arcs.shape)
        # print('true arcs:', true_arcs.shape)

        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-10000] * (l2 - length))
            mask = _model_var(self.model, mask)
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            masks.append(mask.transpose(0, 1))
        length_mask = torch.stack(masks, 0)
        arc_logits = self.arc_logits + length_mask

        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1)

        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        b, l1, d = output_logits.size()
        true_rels = _model_var(self.model, pad_sequence(true_rels, padding=-1, dtype=np.int64))

        rel_loss = F.cross_entropy(
            output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1)

        arc_loss = torch.mean(arc_loss * part_masks.view(b * l1))
        rel_loss = torch.mean(rel_loss * part_masks.view(b * l1))

        loss = arc_loss + rel_loss
        # print("loss: ", loss, "arc_loss: ", arc_loss, "rel_loss: ", rel_loss)

        return loss

    def compute_accuracy(self, true_arcs, true_rels):
        b, l1, l2 = self.arc_logits.size()
        pred_arcs = self.arc_logits.data.max(2)[1].cpu()
        index_true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        arc_correct = pred_arcs.eq(true_arcs).cpu().sum()


        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][arcs[i]])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        pred_rels = output_logits.data.max(2)[1].cpu()
        true_rels = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        label_correct = pred_rels.eq(true_rels).cpu().sum()

        total_arcs = b * l1 - np.sum(true_arcs.cpu().numpy() == -1)

        return arc_correct, label_correct, total_arcs
