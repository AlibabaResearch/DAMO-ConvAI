#coding=utf8
import copy, math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def lens2mask(lens):
    bsize = lens.numel()
    max_len = lens.max()
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks

def lens2mask2(lens,max_len):
    bsize = lens.numel()
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks

def mask2matrix(mask):
    col_mask, row_mask = mask.unsqueeze(-1), mask.unsqueeze(-2)
    return col_mask & row_mask

def tile(x, count, dim=0):
    """
        Tiles x on dimension dim count times.
        E.g. [1, 2, 3], count=2 ==> [1, 1, 2, 2, 3, 3]
            [[1, 2], [3, 4]], count=3, dim=1 ==> [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
        Different from torch.repeat
    """
    if x is None:
        return x
    elif type(x) in [list, tuple]:
        return type(x)([tile(each, count, dim) for each in x])
    else:
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.contiguous().view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x

def rnn_wrapper(encoder, inputs, lens, cell='lstm'):
    """
        @args:
            encoder(nn.Module): rnn series bidirectional encoder, batch_first=True
            inputs(torch.FloatTensor): rnn inputs, [bsize x max_seq_len x in_dim]
            lens(torch.LongTensor): seq len for each sample, allow length=0, padding with 0-vector, [bsize]
        @return:
            out(torch.FloatTensor): output of encoder, bsize x max_seq_len x hidden_dim*2
            hidden_states([tuple of ]torch.FloatTensor): final hidden states, num_layers*2 x bsize x hidden_dim
    """
    # rerank according to lens and remove empty inputs
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_num, total_num = torch.sum(sorted_lens > 0).item(), sorted_lens.size(0)
    sort_key = sort_key[:nonzero_num]
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key)
    # forward non empty inputs    
    packed_inputs = rnn_utils.pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_num].tolist(), batch_first=True)
    packed_out, sorted_h = encoder(packed_inputs)  # bsize x srclen x dim
    sorted_out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    if cell.upper() == 'LSTM':
        sorted_h, sorted_c = sorted_h
    # rerank according to sort_key
    out_shape = list(sorted_out.size())
    out_shape[0] = total_num
    out = sorted_out.new_zeros(*out_shape).scatter_(0, sort_key.unsqueeze(-1).unsqueeze(-1).repeat(1, *out_shape[1:]), sorted_out)
    h_shape = list(sorted_h.size())
    h_shape[1] = total_num
    h = sorted_h.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_h)
    if cell.upper() == 'LSTM':
        c = sorted_c.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_c)
        return out, (h.contiguous(), c.contiguous())
    return out, h.contiguous()

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, q_size, kv_size, output_size, num_heads=8, bias=True, feat_drop=0.2, attn_drop=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = int(num_heads)
        self.hidden_size = hidden_size
        assert self.hidden_size % self.num_heads == 0, 'Head num %d must be divided by hidden size %d' % (num_heads, hidden_size)
        self.d_k = self.hidden_size // self.num_heads
        self.feat_drop = nn.Dropout(p=feat_drop)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.W_q = nn.Linear(q_size, self.hidden_size, bias=bias)
        self.W_k = nn.Linear(kv_size, self.hidden_size, bias=False)
        self.W_v = nn.Linear(kv_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.hidden_size, output_size, bias=bias)

    def forward(self, hiddens, query_hiddens, mask=None):
        ''' @params:
                hiddens : encoded sequence representations, bsize x seqlen x hidden_size
                query_hiddens : bsize [x tgtlen ]x hidden_size
                mask : length mask for hiddens, ByteTensor, bsize x seqlen
            @return:
                context : bsize x[ tgtlen x] hidden_size
        '''
        remove_flag = False
        if query_hiddens.dim() == 2:
            query_hiddens, remove_flag = query_hiddens.unsqueeze(1), True
        Q, K, V = self.W_q(self.feat_drop(query_hiddens)), self.W_k(self.feat_drop(hiddens)), self.W_v(self.feat_drop(hiddens))
        Q, K, V = Q.reshape(-1, Q.size(1), 1, self.num_heads, self.d_k), K.reshape(-1, 1, K.size(1), self.num_heads, self.d_k), V.reshape(-1, 1, V.size(1), self.num_heads, self.d_k)
        e = (Q * K).sum(-1) / math.sqrt(self.d_k) # bsize x tgtlen x seqlen x num_heads
        if mask is not None:
            e = e + ((1 - mask.float()) * (-1e20)).unsqueeze(1).unsqueeze(-1)
        a = torch.softmax(e, dim=2)
        concat = (a.unsqueeze(-1) * V).sum(dim=2).reshape(-1, query_hiddens.size(1), self.hidden_size)
        context = self.W_o(concat)
        if remove_flag:
            return context.squeeze(dim=1), a.mean(dim=-1).squeeze(dim=1)
        else:
            return context, a.mean(dim=-1)

class PoolingFunction(nn.Module):
    """ Map a sequence of hidden_size dim vectors into one fixed size vector with dimension output_size """
    def __init__(self, hidden_size=256, output_size=256, bias=True, method='attentive-pooling'):
        super(PoolingFunction, self).__init__()
        assert method in ['mean-pooling', 'max-pooling', 'attentive-pooling']
        self.method = method
        if self.method == 'attentive-pooling':
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=bias)
            )
        self.mapping_function = nn.Sequential(nn.Linear(hidden_size, output_size, bias=bias), nn.Tanh()) \
            if hidden_size != output_size else lambda x: x

    def forward(self, inputs, mask=None):
        """ @args:
                inputs(torch.FloatTensor): features, batch_size x seq_len x hidden_size
                mask(torch.BoolTensor): mask for inputs, batch_size x seq_len
            @return:
                outputs(torch.FloatTensor): aggregate seq_len dim for inputs, batch_size x output_size
        """
        if self.method == 'max-pooling':
            outputs = inputs.masked_fill(~ mask.unsqueeze(-1), -1e8)
            outputs = outputs.max(dim=1)[0]
        elif self.method == 'mean-pooling':
            mask_float = mask.float().unsqueeze(-1)
            outputs = (inputs * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        elif self.method == 'attentive-pooling':
            e = self.attn(inputs).squeeze(-1)
            e = e + (1 - mask.float()) * (-1e20)
            a = torch.softmax(e, dim=1).unsqueeze(1)
            outputs = torch.bmm(a, inputs).squeeze(1)
        else:
            raise ValueError('[Error]: Unrecognized pooling method %s !' % (self.method))
        outputs = self.mapping_function(outputs)
        return outputs

class FFN(nn.Module):

    def __init__(self, input_size):
        super(FFN, self).__init__()
        self.input_size = input_size
        self.feedforward = nn.Sequential(
            nn.Linear(self.input_size, self.input_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size * 4, self.input_size)
        )
        self.layernorm = nn.LayerNorm(self.input_size)

    def forward(self, inputs):
        return self.layernorm(inputs + self.feedforward(inputs))

class Registrable(object):
    """
    A class that collects all registered components,
    adapted from `common.registrable.Registrable` from AllenNLP
    """
    registered_components = dict()

    @staticmethod
    def register(name):
        def register_class(cls):
            if name in Registrable.registered_components:
                raise RuntimeError('class %s already registered' % name)

            Registrable.registered_components[name] = cls
            return cls

        return register_class

    @staticmethod
    def by_name(name):
        return Registrable.registered_components[name]

class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
