import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ffn_output, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, d_ffn_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.norm(sublayer(x)))
        # return self.dropout(sublayer(self.norm(x)))
        # return x + self.dropout(sublayer(self.norm(x)))

def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])

def relative_attention_logits(query, key, relation):
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # qk_matmul is [batch, heads, num queries, num kvs]
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))

    # q_t is [batch, num queries, heads, depth]
    q_t = query.permute(0, 2, 1, 3)

    # r_t is [batch, num queries, depth, num kvs]
    r_t = relation.transpose(-2, -1)

    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    q_tr_t_matmul = torch.matmul(q_t, r_t)

    # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

    # [batch, heads, num queries, num kvs]
    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

def relative_attention_values(weight, value, relation):
    # In this version, relation vectors are shared across heads.
    # weight: [batch, heads, num queries, num kvs].
    # value: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # wv_matmul is [batch, heads, num queries, depth]
    wv_matmul = torch.matmul(weight, value)

    # w_t is [batch, num queries, heads, num kvs]
    w_t = weight.permute(0, 2, 1, 3)

    #   [batch, num queries, heads, num kvs]
    # * [batch, num queries, num kvs, depth]
    # = [batch, num queries, heads, depth]
    w_tr_matmul = torch.matmul(w_t, relation)

    # w_tr_matmul_t is [batch, heads, num queries, depth]
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

    return wv_matmul + w_tr_matmul_t

def attention_with_relations(query, key, value, relation_k, relation_v, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn_map = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn_map)
    return relative_attention_values(p_attn, value, relation_v), p_attn_map


class MultiHeadedAttentionWithRelations(nn.Module):
    def __init__(self, d_input, d_model, head_num=16, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithRelations, self).__init__()
        assert d_model % head_num == 0
        self.d_k = d_model // head_num
        self.head_num = head_num
        self.linears = clones(lambda: nn.Linear(d_input, d_model), 4)
        self.linear_proj = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, relation_k, relation_v, mask=None):
        # query shape: [batch, num queries, d_model]
        # key shape: [batch, num kv, d_model]
        # value shape: [batch, num kv, d_model]
        # relations_k shape: [batch, num queries, num kv, (d_model // h)]
        # relations_v shape: [batch, num queries, num kv, (d_model // h)]
        # mask shape: [batch, num queries, num kv]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: [batch, heads, num queries, depth]
        x, self.attn_map = attention_with_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            mask=mask,
            dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.head_num * self.d_k)
        return self.linear_proj(x)

class ExplicitEncoder(nn.Module):   
    "Attentio with explicit relation"
    def __init__(self, d_input, d_output, dropout=0.1):
        super(ExplicitEncoder, self).__init__()
        num_relation = 6
        head_num = 16
        d_ffn = d_output
        self.d_input = d_input
        d_model = d_input // 8
        self.d_model = d_model
        self.self_attn = MultiHeadedAttentionWithRelations(d_input, d_model, head_num, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ffn)
        """
        self.sublayer = clones(lambda: SublayerConnection(d_model, dropout), 2)
        self.add_with_norm_1 = self.sublayer[0]
        self.add_with_norm_2 = self.sublayer[1]
        """
        self.add_with_norm_1 = SublayerConnection(d_model, dropout)
        self.add_with_norm_2 = SublayerConnection(d_ffn, dropout)
        self.relation_k_emb = nn.Embedding(num_relation, self.self_attn.d_k)
        self.relation_v_emb = nn.Embedding(num_relation, self.self_attn.d_k)

    @staticmethod
    def compute_relation(batch_size, all_len, question, header):
        relation = torch.zeros((batch_size, all_len, all_len))
        def is_ngram_match(s1, s2, n):
            vis = set()
            for i in range(len(s1) - n + 1):
                vis.add(s1[i:i+n])
            for i in range(len(s2) - n + 1):
                if s2[i:i+n] in vis:
                    return True
            return False
        
        q_len = len(question[0])
        h_len = len(header[0])
        """
        q_len = len(question)
        h_len = len(header)
        """
        for b in range(batch_size):
            for i in range(q_len):
                q = question[b][i].lower()
                idx = i
                for j in range(h_len):
                    h = header[b][j].lower()
                    jdx = j + q_len
                    # relation 1: extract match
                    if q == h:
                        relation[b][idx][jdx] = 1
                        relation[b][jdx][idx] = 1
                    
                    # other relation:
                    for n in [3, 4, 5, 6]:
                        if is_ngram_match(q, h, n):
                            relation[b][idx][jdx] = n - 1
                            relation[b][jdx][idx] = n - 1
        return relation



    def forward(self, wemb_n, wemb_new_h, question, header):
        x = torch.cat([wemb_n, wemb_new_h], dim=1)
        batch_size, all_len, _ = x.shape
        relation = self.compute_relation(batch_size, all_len, question, header).long().cuda()
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)
        x = self.add_with_norm_1(x, lambda x: self.self_attn(x, x, x, relation_k, relation_v))
        x = self.add_with_norm_2(x, self.ffn)
        
        question_len = wemb_n.size()[1]
        question_explicate = x[:,:question_len,:]
        header_explicate = x[:,question_len:,:]
        return question_explicate, header_explicate
