# -*- encoding:utf-8 -*-
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .layer_norm import LayerNorm
from .position_ffn import PositionwiseFeedForward
from .multi_headed_attn import MultiHeadedAttention
from ..encoder import Encoder as Encoder2

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class oldestTransformerLayer(nn.Module):  #oldest
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, num_layers, input_size, state_size):
        super(TransformerLayer, self).__init__()
        #num_layers is no useful
        print('input_size',input_size, type(input_size))
        print('state_size',state_size, type(state_size))
        input_size = int(input_size)
        
        heads_num = -1
        for i in range(4, 10):
            if state_size % i == 0:
                heads_num = i
        assert heads_num != -1

        self.first_linear = nn.Linear(input_size, state_size)

        self.self_attn = MultiHeadedAttention(state_size, heads_num)
        self.layer_norm_1 = LayerNorm(state_size)
        self.feed_forward = PositionwiseFeedForward(state_size, state_size)
        self.layer_norm_2 = LayerNorm(state_size)

        self._1_self_attn = MultiHeadedAttention(state_size, heads_num)
        self._1_layer_norm_1 = LayerNorm(state_size)
        self._1_feed_forward = PositionwiseFeedForward(state_size, state_size)
        self._1_layer_norm_2 = LayerNorm(state_size)
        self._2_self_attn = MultiHeadedAttention(state_size, heads_num)
        self._2_layer_norm_1 = LayerNorm(state_size)
        self._2_feed_forward = PositionwiseFeedForward(state_size, state_size)
        self._2_layer_norm_2 = LayerNorm(state_size)
        '''self._3_self_attn = MultiHeadedAttention(state_size, heads_num)
        self._3_layer_norm_1 = LayerNorm(state_size)
        self._3_feed_forward = PositionwiseFeedForward(state_size, state_size)
        self._3_layer_norm_2 = LayerNorm(state_size)
        self._4_self_attn = MultiHeadedAttention(state_size, heads_num)
        self._4_layer_norm_1 = LayerNorm(state_size)
        self._4_feed_forward = PositionwiseFeedForward(state_size, state_size)
        self._4_layer_norm_2 = LayerNorm(state_size)'''

        self.X = Encoder2(num_layers, input_size, state_size)

        '''self.cell_memories_linear = nn.Linear(1, input_size)
        self.hidden_states_linear = nn.Linear(1, input_size)'''

        self.last_linear = nn.Linear(state_size*2, state_size)
        

    def forward(self, sequence, embedder, dropout_amount=0):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batch_size = 1
        seq_length = len(sequence)
        hidden = []

        for token in sequence:
            hidden.append(embedder(token).unsqueeze(0).unsqueeze(0))
        
        '''hidden.append(self.cell_memories_linear( torch.ones([1,1,1]).cuda() ))
        hidden.append(self.hidden_states_linear( torch.ones([1,1,1]).cuda() ))'''

        hidden = torch.cat(hidden, 1)
        mask = torch.zeros([batch_size, 1, seq_length, seq_length]).cuda()

        hidden = F.dropout(gelu(self.first_linear(hidden)), p=dropout_amount)

        inter = F.dropout(self._1_self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self._1_layer_norm_1(inter + hidden)
        output = F.dropout(self._1_feed_forward(inter), p=dropout_amount)
        hidden = self._1_layer_norm_2(output + inter)
        
        inter = F.dropout(self._2_self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self._2_layer_norm_1(inter + hidden)
        output = F.dropout(self._2_feed_forward(inter), p=dropout_amount)
        hidden = self._2_layer_norm_2(output + inter)
        
        '''inter = F.dropout(self._3_self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self._3_layer_norm_1(inter + hidden)
        output = F.dropout(self._3_feed_forward(inter), p=dropout_amount)
        hidden = self._3_layer_norm_2(output + inter)
        
        inter = F.dropout(self._4_self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self._4_layer_norm_1(inter + hidden)
        output = F.dropout(self._4_feed_forward(inter), p=dropout_amount)
        hidden = self._4_layer_norm_2(output + inter)'''


        inter = F.dropout(self.self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        output = F.dropout(self.feed_forward(inter), p=dropout_amount)

        final_outputs = []

        #assert len(sequence) + 2 == output.size()[1]

        for i in range(len(sequence)):
            x = output[0,i,]
            final_outputs.append(x)

        #cell_memories = [final_outputs[-2]]
        #hidden_states = [final_outputs[-1]]
        #final_outputs = final_outputs[:-2]
        
        #return (cell_memories, hidden_states), final_outputs

        x, final_outputs2 = self.X(sequence, embedder, dropout_amount)

        #print('<', final_outputs[0].mean().item(), final_outputs[0].std().item(), '>', '<', final_outputs2[0].mean().item(), final_outputs2[0].std().item(), '>')

        for i in range(len(sequence)):
            final_outputs[i] = F.dropout(gelu(self.last_linear(torch.cat( [final_outputs2[i], final_outputs[i]]))), p=dropout_amount)

        

        return x, final_outputs

class NewNewTransformerLayer(nn.Module):  #Newnew
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, num_layers, input_size, _state_size):
        super(TransformerLayer, self).__init__()
        #num_layers is no useful
        print('input_size', input_size, type(input_size))
        print('_state_size', _state_size, type(_state_size))
        if _state_size == 650:
            state_size = 324
        else:
            state_size = _state_size//2
        input_size = int(input_size)
        
        heads_num = -1
        for i in range(4, 10):
            if state_size % i == 0:
                heads_num = i
        assert heads_num != -1

        self.first_linear = nn.Linear(input_size, state_size)

        self.self_attn = MultiHeadedAttention(state_size, heads_num)
        self.layer_norm_1 = LayerNorm(state_size)
        self.feed_forward = PositionwiseFeedForward(state_size, state_size)
        self.layer_norm_2 = LayerNorm(state_size)

        self.cell_memories_linear = nn.Linear(state_size, state_size)
        self.hidden_states_linear = nn.Linear(state_size, state_size)

        self.X = Encoder2(num_layers, input_size, _state_size-state_size)

    def forward(self, sequence, embedder, dropout_amount=0):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batch_size = 1
        seq_length = len(sequence)
        hidden = []

        for token in sequence:
            hidden.append(embedder(token).unsqueeze(0).unsqueeze(0))
        

        hidden = torch.cat(hidden, 1)
        mask = torch.zeros([batch_size, 1, seq_length, seq_length]).cuda()

        hidden = F.dropout(gelu(self.first_linear(hidden)), p=dropout_amount)

        inter = F.dropout(self.self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self.layer_norm_1(inter + hidden)
        output = F.dropout(self.feed_forward(inter), p=dropout_amount)
        output = self.layer_norm_2(output + inter)

        final_outputs = []

        assert len(sequence) == output.size()[1]

        for i in range(len(sequence)):
            x = output[0,i,]
            final_outputs.append(x)

        x = output[0].mean(dim=0)
        cell_memories = self.cell_memories_linear(x)
        cell_memories = [F.dropout(gelu(cell_memories), p=dropout_amount)]
        hidden_states = self.hidden_states_linear(x)
        hidden_states = [F.dropout(gelu(hidden_states), p=dropout_amount)]

        '''print('hidden_states[0]', hidden_states[0].size())
        print('cell_memories[0]', cell_memories[0].size())
        print('final_outputs[0]', final_outputs[0].size())'''
        
        x, final_outputs2 = self.X(sequence, embedder, dropout_amount)

        cell_memories2 = x[0]
        hidden_states2 = x[1]

        for i in range(len(sequence)):
            final_outputs[i] = torch.cat( [final_outputs[i], final_outputs2[i]])
        
        cell_memories[0] = torch.cat( [cell_memories[0], cell_memories2[0]])
        hidden_states[0] = torch.cat( [hidden_states[0], hidden_states2[0]])

        '''print('hidden_states[0]', hidden_states[0].size())
        print('cell_memories[0]', cell_memories[0].size())
        print('final_outputs[0]', final_outputs[0].size())'''

        return (cell_memories, hidden_states), final_outputs

class TransformerLayer(nn.Module):  #New
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, num_layers, input_size, state_size):
        super(TransformerLayer, self).__init__()
        #num_layers is no useful
        print('input_size',input_size, type(input_size))
        print('state_size',state_size, type(state_size))
        input_size = int(input_size)
        
        heads_num = -1
        for i in range(4, 10):
            if state_size % i == 0:
                heads_num = i
        assert heads_num != -1

        self.first_linear = nn.Linear(input_size, state_size)

        self.self_attn = MultiHeadedAttention(state_size, heads_num)
        self.layer_norm_1 = LayerNorm(state_size)
        self.feed_forward = PositionwiseFeedForward(state_size, state_size)
        self.layer_norm_2 = LayerNorm(state_size)

        self.X = Encoder2(num_layers, input_size, state_size)
        self.last_linear = nn.Linear(state_size*2, state_size)

    def forward(self, sequence, embedder, dropout_amount=0):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batch_size = 1
        seq_length = len(sequence)
        hidden = []

        for token in sequence:
            hidden.append(embedder(token).unsqueeze(0).unsqueeze(0))
        

        hidden = torch.cat(hidden, 1)
        mask = torch.zeros([batch_size, 1, seq_length, seq_length]).cuda()

        hidden = F.dropout(gelu(self.first_linear(hidden)), p=dropout_amount)

        inter = F.dropout(self.self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self.layer_norm_1(inter + hidden)
        output = F.dropout(self.feed_forward(inter), p=dropout_amount)
        output = self.layer_norm_2(output + inter)

        final_outputs = []

        assert len(sequence) == output.size()[1]

        for i in range(len(sequence)):
            x = output[0,i,]
            final_outputs.append(x)

        x, final_outputs2 = self.X(sequence, embedder, dropout_amount)

        for i in range(len(sequence)):
            #final_outputs[i] = F.dropout(final_outputs2[i]+final_outputs[i], p=dropout_amount)
            final_outputs[i] = F.dropout(gelu(self.last_linear(torch.cat( [final_outputs2[i], final_outputs[i]]))), p=dropout_amount)

        return x, final_outputs

class Old2only2TransformerLayer(nn.Module):    #Old2   2020 05 24 01.47
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, num_layers, input_size, state_size):
        super(TransformerLayer, self).__init__()
        #num_layers is no useful
        print('input_size',input_size, type(input_size))
        print('state_size',state_size, type(state_size))
        input_size = int(input_size)
        
        heads_num = -1
        for i in range(4, 10):
            if state_size % i == 0:
                heads_num = i
        assert heads_num != -1

        self.first_linear = nn.Linear(input_size, state_size)

        self.self_attn = MultiHeadedAttention(state_size, heads_num)
        self.layer_norm_1 = LayerNorm(state_size)
        self.feed_forward = PositionwiseFeedForward(state_size, state_size)
        self.layer_norm_2 = LayerNorm(state_size)

        #self.cell_memories_linear = nn.Linear(state_size, state_size)
        #self.hidden_states_linear = nn.Linear(state_size, state_size)

        #self.last_linear = nn.Linear(state_size+state_size, state_size)

    def forward(self, sequence, embedder, dropout_amount=0):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batch_size = 1
        seq_length = len(sequence)
        hidden = []

        for token in sequence:
            hidden.append(embedder(token).unsqueeze(0).unsqueeze(0))

        hidden = torch.cat(hidden, 1)
        mask = torch.zeros([batch_size, 1, seq_length, seq_length]).cuda()

        hidden = F.dropout(gelu(self.first_linear(hidden)), p=dropout_amount)

        inter = F.dropout(self.self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self.layer_norm_1(inter + hidden)
        output = F.dropout(self.feed_forward(inter), p=dropout_amount)
        output = self.layer_norm_2(output + inter)

        final_outputs = []

        assert len(sequence) == output.size()[1]

        #output = F.dropout(gelu( self.last_linear(torch.cat([output, hidden], dim=-1)) ), p=dropout_amount)

        for i in range(len(sequence)):
            x = output[0,i,] 
            final_outputs.append(x)

        #x = output[0].mean(dim=0)
        #cell_memories = self.cell_memories_linear(x)
        #cell_memories = [F.dropout(gelu(cell_memories), p=dropout_amount)]
        #hidden_states = self.hidden_states_linear(x)
        #hidden_states = [F.dropout(gelu(hidden_states), p=dropout_amount)]
        
        
        #return (cell_memories, hidden_states), final_outputs
        return None, final_outputs

class OldTransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, num_layers, input_size, state_size):
        super(TransformerLayer, self).__init__()
        #num_layers is no useful
        print('input_size',input_size, type(input_size))
        print('state_size',state_size, type(state_size))
        input_size = int(input_size)
        
        heads_num = -1
        for i in range(4, 10):
            if state_size % i == 0:
                heads_num = i
        assert heads_num != -1

        self.first_linear = nn.Linear(input_size, state_size)

        self.self_attn = MultiHeadedAttention(state_size, heads_num)
        self.layer_norm_1 = LayerNorm(state_size)
        self.feed_forward = PositionwiseFeedForward(state_size, state_size)
        self.layer_norm_2 = LayerNorm(state_size)

        self.cell_memories_linear = nn.Linear(1, input_size)
        self.hidden_states_linear = nn.Linear(1, input_size)
        

    def forward(self, sequence, embedder, dropout_amount=0):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batch_size = 1
        seq_length = len(sequence)
        hidden = []

        for token in sequence:
            hidden.append(embedder(token).unsqueeze(0).unsqueeze(0))
        
        hidden.append(self.cell_memories_linear( torch.ones([1,1,1]).cuda() ))
        hidden.append(self.hidden_states_linear( torch.ones([1,1,1]).cuda() ))

        hidden = torch.cat(hidden, 1)
        mask = torch.zeros([batch_size, 1, seq_length+2, seq_length+2]).cuda()

        hidden = F.dropout(gelu(self.first_linear(hidden)), p=dropout_amount)

        inter = F.dropout(self.self_attn(hidden, hidden, hidden, mask, dropout_amount), p=dropout_amount)
        inter = self.layer_norm_1(inter + hidden)
        output = F.dropout(self.feed_forward(inter), p=dropout_amount)
        output = self.layer_norm_2(output + inter)

        final_outputs = []

        assert len(sequence) + 2 == output.size()[1]

        for i in range(len(sequence)+2):
            x = output[0,i,]
            final_outputs.append(x)

        cell_memories = [final_outputs[-2]]
        hidden_states = [final_outputs[-1]]
        final_outputs = final_outputs[:-2]
        
        return (cell_memories, hidden_states), final_outputs
