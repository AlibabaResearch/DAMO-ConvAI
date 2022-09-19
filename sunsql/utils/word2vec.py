#coding=utf8

from embeddings import GloveEmbedding
import numpy as np
from utils.constants import PAD
import torch, random

class Word2vecUtils():

    def __init__(self):
        super(Word2vecUtils, self).__init__()
        self.word_embed = GloveEmbedding('common_crawl_48', d_emb=300)
        self.initializer = lambda: np.random.normal(size=300).tolist()

    def load_embeddings(self, module, vocab, device='cpu'):
        """ Initialize the embedding with glove and char embedding
        """
        emb_size = module.weight.data.size(-1)
        assert emb_size == 300, 'Embedding size is not 300, cannot be initialized by GLOVE'
        outliers = 0
        for word in vocab.word2id:
            if word == PAD: # PAD symbol is always 0-vector
                module.weight.data[vocab[PAD]] = torch.zeros(emb_size, dtype=torch.float, device=device)
                continue
            word_emb = self.word_embed.emb(word, default='none')
            if word_emb[0] is None: # oov
                word_emb = self.initializer()
                outliers += 1
            module.weight.data[vocab[word]] = torch.tensor(word_emb, dtype=torch.float, device=device)
        return 1 - outliers / float(len(vocab))

    def emb(self, word):
        word_emb = self.word_embed.emb(word, default='none')
        if word_emb[0] is None:
            return None
        else:
            return word_emb
