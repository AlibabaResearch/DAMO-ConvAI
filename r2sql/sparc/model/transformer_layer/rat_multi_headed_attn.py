import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class RatMultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size, heads_num, relationship_number):
        super(RatMultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        
        if relationship_number != -1:
            assert relationship_number == self.per_head_size
            self.relationship_K_parameter = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(heads_num, self.per_head_size)))
            self.relationship_V_parameter = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(heads_num, self.per_head_size)))
            


        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, relationship_matrix, dropout):
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]
        


        relationship_matrix = relationship_matrix.unsqueeze(3).repeat(1, 1, 1, heads_num, 1)
        kk = self.relationship_K_parameter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, seq_length, 1, 1)
        vv = self.relationship_V_parameter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, seq_length, 1, 1)

        relationship_K = relationship_matrix * kk
        relationship_V = relationship_matrix * vv
        relationship_K = relationship_K.transpose(1, 3)
        relationship_V = relationship_V.transpose(1, 3)

        query = query.unsqueeze(3).repeat(1, 1, 1, seq_length, 1)
        key = key.unsqueeze(2).repeat(1, 1, seq_length, 1, 1) 
        
        scores = (query*(key+relationship_K)).sum(dim=-1)
        scores = scores / math.sqrt(float(per_head_size))
 
        probs = nn.Softmax(dim=-1)(scores)
        probs = probs.unsqueeze(4).repeat(1, 1, 1, 1, per_head_size)
        value = value.unsqueeze(2).repeat(1, 1, seq_length, 1, 1)
        
        output = (probs*(value+relationship_V)).sum(dim=-2)
        output = unshape(output)
        output = self.final_linear(output)
        return output
