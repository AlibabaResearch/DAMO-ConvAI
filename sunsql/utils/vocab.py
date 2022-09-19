#coding=utf8
from utils.constants import PAD, UNK, BOS, EOS

class Vocab():

    def __init__(self, padding=False, unk=False, boundary=False, min_freq=1,
            filepath=None, iterable=None, default=UNK, specials=[]):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        self.default = default # if default is None, ensure that no oov words
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK
        if boundary:
            idx = len(self.word2id)
            self.word2id[BOS], self.id2word[idx] = idx, BOS
            self.word2id[EOS], self.id2word[idx + 1] = idx + 1, EOS
        for w in specials:
            if w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w], self.id2word[idx] = idx, w
        if filepath is not None:
            self.from_filepath(filepath, min_freq=min_freq)
        elif iterable is not None:
            self.from_iterable(iterable)
        assert (self.default is None) or (self.default in self.word2id)

    def from_filepath(self, filepath, min_freq=1):
        with open(filepath, 'r', encoding='utf-8') as inf:
            for line in inf:
                line = line.strip()
                if line == '': continue
                line = line.split('\t') # ignore count or frequency
                if len(line) == 1:
                    word, freq = line[0], min_freq
                else:
                    assert len(line) == 2
                    word, freq = line
                word = word.lower()
                if word not in self.word2id and int(freq) >= min_freq:
                    idx = len(self.word2id)
                    self.word2id[word] = idx
                    self.id2word[idx] = word

    def from_iterable(self, iterable):
        for item in iterable:
            if item not in self.word2id:
                idx = len(self.word2id)
                self.word2id[item] = idx
                self.id2word[idx] = item

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        """ If self.default is None, it means we do not allow out of vocabulary token;
        If self.default is not None, we get the idx of self.default if key does not exist.
        """
        if self.default is None:
            return self.word2id[key]
        else:
            return self.word2id.get(key, self.word2id[self.default])
