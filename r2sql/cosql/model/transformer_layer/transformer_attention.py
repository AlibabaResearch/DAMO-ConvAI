import torch.nn as nn
import torch
import math
import torch.nn.functional as F

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .layer_norm import LayerNorm
from .position_ffn import PositionwiseFeedForward
from .multi_headed_attn import MultiHeadedAttention
from .rat_transformer_layer import RATTransoformer

class TransformerAttention(nn.Module):
    def __init__(self, input_size, state_size):
        super(TransformerAttention, self).__init__()
        input_size = int(input_size)
        assert input_size == state_size
        self.relationship_number = 10
        self.rat_transoformer = RATTransoformer(input_size, state_size, relationship_number=self.relationship_number)
        

    def forward(self, utterance_states, schema_states, input_sequence, unflat_sequence, input_schema, schema, dropout_amount=0):
        """
        Args:
            utterance_states : [utterance_len x emb_size]
            schema_states : [schema_len x emb_size]
            input_sequence : utterance_len
            input_schema : schema_len

        Returns:
            utterance_output: [emb_size x utterance_len]
            schema_output: [emb_size x schema_len]
        """

        assert utterance_states.size()[1] == 350
        utterance_states = utterance_states[:,:300]

        print('input_sequence', input_sequence)
        #print('input_schema.column_names_surface_form', input_schema.column_names_surface_form)

        if True:
            #print('foreign_keys', schema['foreign_keys'])
            #print('input_schema.column_names_surface_form', input_schema.column_names_surface_form)

            utterance_len = utterance_states.size()[0]
            schema_len = schema_states.size()[0]
            #print(utterance_len, schema_len)
            assert len(input_sequence) == utterance_len
            assert len(input_schema.column_names_surface_form) == schema_len

            relationship_matrix = torch.zeros([ utterance_len+schema_len, utterance_len+schema_len,self.relationship_number])

            def get_name(s):
                if s == '*':
                    return '', ''
                table_name, column_num = s.split('.')

                if column_num == '*':
                    column_num = ''
                return table_name, column_num

            def is_ngram_match(s1, s2, n):
                vis = set()
                for i in range(len(s1)-n+1):
                    vis.add( s1[i:i+n] )
                for i in range(len(s2)-n+1):
                    if s2[i:i+n] in vis:
                        return True
                return False

            ''' 0.      same table
                1.      foreign key
                2.      word and table: exact math    
                3-5.    word and table: n-gram 3-5
                6.      word and column: exact math
                7-9.  word and column: n-gram 3-5'''
            for i in range(schema_len):
                table_i, column_i = get_name(input_schema.column_names_surface_form[i])
                idx = i+utterance_len
                for j in range(schema_len):
                    table_j, column_j = get_name(input_schema.column_names_surface_form[j])
                    jdx = j+utterance_len

                    if table_i == table_j:
                        relationship_matrix[idx][jdx][0] = 1
            
            for i, j in schema['foreign_keys']:
                idx = i+utterance_len
                jdx = j+utterance_len

                
                relationship_matrix[idx][jdx][1] = 1
                relationship_matrix[jdx][idx][1] = 1

                #print(idx, jdx)
                #print(input_schema.column_names_surface_form)
                #print(input_schema.column_names_surface_form[i], input_schema.column_names_surface_form[j])
                
            
            for i in range(schema_len):
                table, column = get_name(input_schema.column_names_surface_form[i])
                idx = i+utterance_len
                for j in range(utterance_len):
                    word = input_sequence[j].lower()
                    jdx = j

                    #word and table
                    no_math = True
                    if word == table:
                        relationship_matrix[idx][jdx][2] = 1
                        relationship_matrix[jdx][idx][2] = 1
                    
                    for n in [5,4,3]:
                        if no_math and is_ngram_match(word, table, n):
                            relationship_matrix[idx][jdx][n] = 1
                            relationship_matrix[jdx][idx][n] = 1

                    #word and column
                    no_math = True
                    if word == column:
                        relationship_matrix[idx][jdx][6] = 1
                        relationship_matrix[jdx][idx][6] = 1
                    
                    for n in [5,4,3]:
                        if no_math and is_ngram_match(word, column, n):
                            relationship_matrix[idx][jdx][4+n] = 1
                            relationship_matrix[jdx][idx][4+n] = 1
            
            
            #print(relationship_matrix)
            #print(unflat_sequence)
            
            # start = 0
            # for i in unflat_sequence:
            #     start = len(i) + start
            #     relationship_matrix[0:start][0:start][:] *= 0.98
            
            relationship_matrix = relationship_matrix.unsqueeze(0).cuda() #[1, all_len, all_len, 17]
            
            """
            print(relationship_matrix.shape)
            matrix = torch.zeros_like(relationship_matrix[:,:,:,0]).cuda()
            print(matrix.shape)
            for i in range(relationship_matrix.shape[-1]):
                matrix += relationship_matrix[:,:,:,i]
                #matrix = matrix.squeeze(0).cpu().numpy()
                label = input_sequence + input_schema.column_names_surface_form
            matrix = matrix.squeeze(0).cpu().numpy()
            save_matrix(matrix, label, 0)
            """
            utterance_and_schema_states = torch.cat([utterance_states, schema_states], dim=0).unsqueeze(0) #[1, all_len, emb_size]
            utterance_and_schema_output = self.rat_transoformer(utterance_and_schema_states, relationship_matrix, 0) #[1, all_len, emb_size]
            #print(utterance_and_schema_output.shape)
            utterance_output = utterance_and_schema_output[0,:utterance_len,:]
            schema_output = utterance_and_schema_output[0,utterance_len:,:]

        else:
            utterance_output = utterance_states
            schema_output = schema_states
        utterance_output = torch.transpose(utterance_output, 1, 0)
        schema_output = torch.transpose(schema_output, 1, 0)

        return utterance_output, schema_output


def save_matrix(matrix, label, i):
    """
    matrix: n x n
    label: len(str) == n
    """
    plt.rcParams["figure.figsize"] = (1000,1000)
    plt.matshow(matrix, cmap=plt.cm.Reds)
    plt.xticks(np.arange(len(label)), label, rotation=90, fontsize=16)
    plt.yticks(np.arange(len(label)), label, fontsize=16)
    plt.margins(2)
    plt.subplots_adjust()
    plt.savefig('relation_' + str(i), dpi=300)
    exit()
