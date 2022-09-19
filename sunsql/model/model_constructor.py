#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable, PoolingFunction, SCM
from model.encoder.graph_encoder import *
from model.decoder.sql_parser import *


@Registrable.register('text2sql')
class Text2SQL(nn.Module):
    def __init__(self, args, transition_system):
        super(Text2SQL, self).__init__()
        self.alpha = 1e-3
        self.encoder = Registrable.by_name('encoder_text2sql')(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = Registrable.by_name('decoder_tranx')(args, transition_system)
        self.scm = SCM(args.gnn_hidden_size, bias=True, method='conv')

        
    def forward(self, batch, flag = True):
        """ This function is used during training, which returns the entire training loss
        """
        encodings, mask, gp_loss = self.encoder(batch)
        encodings, semantic_loss = self.scm(encodings, mask=mask, train=True, flag = flag)
        h0 = self.encoder2decoder(encodings, mask=mask)
        loss = self.decoder.score(encodings, mask, h0, batch)
        loss += self.alpha * semantic_loss
            
        return loss, gp_loss


    def parse(self, batch, beam_size):
        """ This function is used for decoding, which returns a batch of [DecodeHypothesis()] * beam_size
        """
        encodings, mask = self.encoder(batch)
        encodings, _ = self.scm(encodings, mask=mask, train=False)
        h0 = self.encoder2decoder(encodings, mask=mask)
        hyps = []
        for i in range(len(batch)):
            """ table_mappings and column_mappings are used to map original database ids to local ids,
            while reverse_mappings perform the opposite function, mapping local ids to database ids
            """
            hyps.append(self.decoder.parse(encodings[i:i+1], mask[i:i+1], h0[i:i+1], batch, beam_size))
        return hyps

    def pad_embedding_grad_zero(self, index=None):
        """ For glove.42B.300d word vectors, gradients for <pad> symbol is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)
