import torch.nn as nn
import torch
import math
import torch.nn.functional as F

import numpy as np

from .layer_norm import LayerNorm
from .position_ffn import PositionwiseFeedForward
from .rat_transformer_layer import RATTransformer

class TransformerAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransformerAttention, self).__init__()
        input_size = int(input_size)
        hidden_size = hidden_size
        self.relation_number = 6
        self.rat_transformer = RATTransformer(input_size, hidden_size, relationship_number=self.relation_number)


    def forward(self, question_emb, header_emb, question, header, dropout_amount=0):
        batch_size = question_emb.size()[0]
        question_len = question_emb.size()[1]
        header_len = header_emb.size()[1]
        all_len = question_len + header_len
        relationship_matrix = torch.zeros((batch_size, all_len, all_len, self.relation_number))
        
        def is_ngram_match(s1, s2, n):
            vis = set()
            for i in range(len(s1) - n + 1):
                vis.add(s1[i:i+n])
            for i in range(len(s2) - n + 1):
                if s2[i:i+n] in vis:
                    return True
            return False
            
        for b in range(batch_size):
            for i in range(header_len):
                h = header[b][i].lower()
                idx = i + question_len
                for j in range(question_len):
                    q = question[b][j].lower()
                    jdx = j

                    if q == h:
                        relationship_matrix[b][idx][jdx][0] = 1
                        relationship_matrix[b][jdx][idx][0] = 1

                    for n in [3, 4, 5, 6]:
                        if is_ngram_match(q, h, n):
                            relationship_matrix[b][idx][jdx][n-2] = 1
                            relationship_matrix[b][jdx][idx][n-2] = 1
        
        relationship_matrix = relationship_matrix.cuda()
        question_and_header_emb = torch.cat([question_emb, header_emb], dim=1)
        explicate_emb = self.rat_transformer(question_and_header_emb, relationship_matrix)
        question_explicate = explicate_emb[:,:question_len,:]
        header_explicate = explicate_emb[:,question_len:,:]
        return question_explicate, header_explicate
                    
                    
                    
         
        
