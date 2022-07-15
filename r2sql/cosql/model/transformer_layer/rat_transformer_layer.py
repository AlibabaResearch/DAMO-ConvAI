# -*- encoding:utf-8 -*-
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .layer_norm import LayerNorm
from .position_ffn import PositionwiseFeedForward
from .rat_multi_headed_attn import RatMultiHeadedAttention

class RATTransoformer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, input_size, state_size, relationship_number):
        super(RATTransoformer, self).__init__()
        input_size = int(input_size)
        assert input_size == state_size
        
        heads_num = 30

        self.self_attn = RatMultiHeadedAttention(state_size, heads_num, relationship_number)
        #self.layer_norm_1 = LayerNorm(state_size)
        self.feed_forward = PositionwiseFeedForward(state_size, state_size)
        #self.layer_norm_2 = LayerNorm(state_size)
        

    def forward(self, hidden, relationship_matrix, dropout_amount=0):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        assert dropout_amount == 0
        batch_size = 1
        seq_length = hidden.size()[1]

        inter = self.self_attn(hidden, hidden, hidden, relationship_matrix, dropout_amount)
        output = self.feed_forward(inter)
       
        # inter = F.dropout(self.self_attn(hidden, hidden, hidden, relationship_matrix, dropout_amount), p=dropout_amount)
        #inter = self.layer_norm_1(inter + hidden)
        # output = F.dropout(self.feed_forward(inter), p=dropout_amount)
        #output = self.layer_norm_2(output + inter)

        return output
