#coding=utf8
import torch
import numpy as np
from utils.example import Example, get_position_ids
from utils.constants import PAD, UNK
from model.model_utils import lens2mask, cached_property
import torch.nn.functional as F

def from_example_list_base(ex_list, device='cpu', train=True):
    """
        question_lens: torch.long, bsize
        questions: torch.long, bsize x max_question_len, include [CLS] if add_cls
        table_lens: torch.long, bsize, number of tables for each example
        table_word_lens: torch.long, number of words for each table name
        tables: torch.long, sum_of_tables x max_table_word_len
        column_lens: torch.long, bsize, number of columns for each example
        column_word_lens: torch.long, number of words for each column name
        columns: torch.long, sum_of_columns x max_column_word_len
    """
    batch = Batch(ex_list, device)
    plm = Example.plm
    pad_idx = Example.word_vocab[PAD] if plm is None else Example.tokenizer.pad_token_id

    question_lens = [len(ex.question) for ex in ex_list]
    batch.question_lens = torch.tensor(question_lens, dtype=torch.long, device=device)
    batch.table_lens = torch.tensor([len(ex.table) for ex in ex_list], dtype=torch.long, device=device)
    table_word_lens = [len(t) for ex in ex_list for t in ex.table]
    batch.table_word_lens = torch.tensor(table_word_lens, dtype=torch.long, device=device)
    batch.column_lens = torch.tensor([len(ex.column) for ex in ex_list], dtype=torch.long, device=device)
    column_word_lens = [len(c) for ex in ex_list for c in ex.column]
    batch.column_word_lens = torch.tensor(column_word_lens, dtype=torch.long, device=device)


    if plm is None: # glove.42B.300d
        questions = [ex.question_id + [pad_idx] * (batch.max_question_len - len(ex.question_id)) for ex in ex_list]
        batch.questions = torch.tensor(questions, dtype=torch.long, device=device)
        tables = [t + [pad_idx] * (batch.max_table_word_len - len(t)) for ex in ex_list for t in ex.table_id]
        batch.tables = torch.tensor(tables, dtype=torch.long, device=device)
        columns = [c + [pad_idx] * (batch.max_column_word_len - len(c)) for ex in ex_list for c in ex.column_id]
        batch.columns = torch.tensor(columns, dtype=torch.long, device=device)
    else:
        # prepare inputs for pretrained models
        batch.inputs = {"input_ids": None, "attention_mask": None, "token_type_ids": None, "position_ids": None}
        input_lens = [len(ex.input_id) for ex in ex_list]
        max_len = max(input_lens)
        input_ids = [ex.input_id + [pad_idx] * (max_len - len(ex.input_id)) for ex in ex_list]
        batch.inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = [[1] * l + [0] * (max_len - l) for l in input_lens]
        batch.inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.float, device=device)
        token_type_ids = [ex.segment_id + [0] * (max_len - len(ex.segment_id)) for ex in ex_list]
        batch.inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long, device=device)
        position_ids = [get_position_ids(ex, shuffle=train) + [0] * (max_len - len(ex.input_id)) for ex in ex_list]
        batch.inputs["position_ids"] = torch.tensor(position_ids, dtype=torch.long, device=device)
        # extract representations after plm, remove [SEP]
        question_mask_plm = [ex.question_mask_plm + [0] * (max_len - len(ex.question_mask_plm)) for ex in ex_list]
        batch.question_mask_plm = torch.tensor(question_mask_plm, dtype=torch.bool, device=device)
        table_mask_plm = [ex.table_mask_plm + [0] * (max_len - len(ex.table_mask_plm)) for ex in ex_list]
        batch.table_mask_plm = torch.tensor(table_mask_plm, dtype=torch.bool, device=device)
        column_mask_plm = [ex.column_mask_plm + [0] * (max_len - len(ex.column_mask_plm)) for ex in ex_list]
        batch.column_mask_plm = torch.tensor(column_mask_plm, dtype=torch.bool, device=device)
        # subword aggregation
        question_subword_lens = [l for ex in ex_list for l in ex.question_subword_len]
        batch.question_subword_lens = torch.tensor(question_subword_lens, dtype=torch.long, device=device)
        table_subword_lens = [l for ex in ex_list for l in ex.table_subword_len]
        batch.table_subword_lens = torch.tensor(table_subword_lens, dtype=torch.long, device=device)
        column_subword_lens = [l for ex in ex_list for l in ex.column_subword_len]
        batch.column_subword_lens = torch.tensor(column_subword_lens, dtype=torch.long, device=device)

    batch.question_unk_mask, batch.table_unk_mask, batch.column_unk_mask = None, None, None
    if not train and plm is None:
        # during evaluation, for words not in vocab but in glove vocab, extract its correpsonding embedding
        word2vec, unk_idx = Example.word2vec, Example.word_vocab[UNK]

        question_unk_mask = (batch.questions == unk_idx).cpu()
        if question_unk_mask.any().item():
            raw_questions = np.array([ex.question + [PAD] * (batch.max_question_len - len(ex.question)) for ex in ex_list], dtype='<U100')
            unk_words = raw_questions[question_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.question_unk_mask = question_unk_mask.masked_scatter_(torch.clone(question_unk_mask), oov_flag).to(device)
                batch.question_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)

        table_unk_mask = (batch.tables == unk_idx).cpu()
        if table_unk_mask.any().item():
            raw_tables = np.array([t + [PAD] * (batch.max_table_word_len - len(t)) for ex in ex_list for t in ex.table], dtype='<U100')
            unk_words = raw_tables[table_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.table_unk_mask = table_unk_mask.masked_scatter_(torch.clone(table_unk_mask), oov_flag).to(device)
                batch.table_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)

        column_unk_mask = (batch.columns == unk_idx).cpu()
        if column_unk_mask.any().item():
            raw_columns = np.array([c + [PAD] * (batch.max_column_word_len - len(c)) for ex in ex_list for c in ex.column], dtype='<U100')
            unk_words = raw_columns[column_unk_mask.numpy()].tolist()
            unk_word_embeddings = [word2vec.emb(w) for w in unk_words]
            oov_flag = torch.tensor([True if e is not None else False for e in unk_word_embeddings], dtype=torch.bool)
            if oov_flag.any().item():
                batch.column_unk_mask = column_unk_mask.masked_scatter_(torch.clone(column_unk_mask), oov_flag).to(device)
                batch.column_unk_embeddings = torch.tensor([e for e in unk_word_embeddings if e is not None], dtype=torch.float, device=device)
    return batch

def from_example_list_text2sql(ex_list, device='cpu', train=True, **kwargs):
    """ New fields: batch.lens, batch.max_len, batch.relations, batch.relations_mask
    """
    batch = from_example_list_base(ex_list, device, train)
    batch.graph = Example.graph_factory.batch_graphs(ex_list, device, train=train, **kwargs)
    if train:
        batch.max_action_num = max([len(ex.tgt_action) for ex in ex_list])
    return batch

class Batch():

    def __init__(self, examples, device='cpu'):
        super(Batch, self).__init__()
        self.examples = examples
        self.device = device

    @classmethod
    def from_example_list(cls, ex_list, device='cpu', train=True, method='text2sql', **kwargs):
        method_dict = {
            "text2sql": from_example_list_text2sql,
        }
        return method_dict[method](ex_list, device, train=train, **kwargs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @cached_property
    def max_question_len(self):
        return torch.max(self.question_lens).item()

    @cached_property
    def max_table_len(self):
        return torch.max(self.table_lens).item()

    @cached_property
    def max_column_len(self):
        return torch.max(self.column_lens).item()

    @cached_property
    def max_table_word_len(self):
        return torch.max(self.table_word_lens).item()

    @cached_property
    def max_column_word_len(self):
        return torch.max(self.column_word_lens).item()

    @cached_property
    def max_question_subword_len(self):
        return torch.max(self.question_subword_lens).item()

    @cached_property
    def max_table_subword_len(self):
        return torch.max(self.table_subword_lens).item()

    @cached_property
    def max_column_subword_len(self):
        return torch.max(self.column_subword_lens).item()

    """ Different types of nodes are seperated instead of concatenated together """
    @cached_property
    def mask(self):
        return torch.cat([self.question_mask, self.table_mask, self.column_mask], dim=1)

    @cached_property
    def question_mask(self):
        return lens2mask(self.question_lens)

    @cached_property
    def table_mask(self):
        return lens2mask(self.table_lens)

    @cached_property
    def column_mask(self):
        return lens2mask(self.column_lens)

    @cached_property
    def table_word_mask(self):
        return lens2mask(self.table_word_lens)

    @cached_property
    def column_word_mask(self):
        return lens2mask(self.column_word_lens)

    @cached_property
    def question_subword_mask(self):
        return lens2mask(self.question_subword_lens)

    @cached_property
    def table_subword_mask(self):
        return lens2mask(self.table_subword_lens)

    @cached_property
    def column_subword_mask(self):
        return lens2mask(self.column_subword_lens)

    def get_frontier_field_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.field2id[e.tgt_action[t].frontier_field])
                # assert self.grammar.id2field[ids[-1]] == e.tgt_action[t].frontier_field
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_prod_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.prod2id[e.tgt_action[t].frontier_prod])
                # assert self.grammar.id2prod[ids[-1]] == e.tgt_action[t].frontier_prod
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_field_type_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.type2id[e.tgt_action[t].frontier_field.type])
                # assert self.grammar.id2type[ids[-1]] == e.tgt_action[t].frontier_field.type
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)
