import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .layer_norm import LayerNorm
from .position_ffn import PositionwiseFeedForward
from .rat_multi_headed_attn import RatMultiHeadedAttention

class RATTransformer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, input_size, hidden_size, relationship_number):
        super(RATTransformer, self).__init__()
        input_size = int(input_size)
        heads_num = 50
        self.self_attn = RatMultiHeadedAttention(input_size, hidden_size, heads_num, relationship_number)
        #self.layer_norm_1 = LayerNorm(state_size)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size)
        #self.layer_norm_2 = LayerNorm(state_size)
        

    def forward(self, hidden, relationship_matrix, dropout_amount=0):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        seq_length = hidden.size()[1]
        #self.layer_norm_1(hidden)
        inter = self.self_attn(hidden, hidden, hidden, relationship_matrix, dropout_amount)
        #self.layer_norm_2(inter)
        output = self.feed_forward(inter)
        # inter = F.dropout(self.self_attn(hidden, hidden, hidden, relationship_matrix, dropout_amount), p=dropout_amount)
        #inter = self.layer_norm_1(inter + hidden)
        # output = F.dropout(self.feed_forward(inter), p=dropout_amount)
        #output = self.layer_norm_2(output + inter)

        return output
