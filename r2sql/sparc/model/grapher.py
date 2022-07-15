import numpy as np
import spacy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

nlp = spacy.load('en_core_web_sm')

class Grapher(nn.Module):
    def __init__(self, input_size=300, output_size=300):
        super(Grapher, self).__init__()
        self.gc_1 = GCN(in_features=input_size, out_features=300)
        self.gcn_drop = nn.Dropout(0.2)
        # self.fc = nn.Linear(128, 300).cuda()
    
    def gen_adj(self, input_sequence):
        self.adj = torch.from_numpy(self.dependency_adj_matrix(' '.join(input_sequence))).cuda()
        

    def dependency_adj_matrix(self, text):
        # https://spacy.io/docs/usage/processing-text
        document = nlp(text)
        seq_len = len(text.split())
        matrix = np.zeros((seq_len, seq_len)).astype('float32')
        
        for token in document:
            if token.i < seq_len:
                matrix[token.i][token.i] = 1
                # https://spacy.io/docs/api/token
                for child in token.children:
                    if child.i < seq_len:
                        matrix[token.i][child.i] = 1
                        matrix[child.i][token.i] = 1
        return matrix

    def forward(self, x):
        gnn_emb = self.gc_1(x, self.adj)
        gnn_emb = F.relu(gnn_emb)
        # gnn_emb = self.gcn_drop(gnn_emb)
        # gnn_emb = self.fc(gnn_emb)
        # gnn_emb = F.relu(gnn_emb)
        return gnn_emb

class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).cuda()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).cuda()
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, hidden, adj):
        # H * W
        # print(self.weight)
        hidden = torch.matmul(hidden, self.weight)
        # process A
        denom = torch.sum(adj, dim=1, keepdim=True) + 1
        # A * H * W
        output = torch.matmul(adj, hidden) / denom
        # print(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
