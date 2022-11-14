#coding=utf8
import os, math
import torch
import torch.nn as nn
from model.model_utils import rnn_wrapper, lens2mask, PoolingFunction
from transformers import AutoModel, AutoConfig

class GraphInputLayer(nn.Module):

    def __init__(self, embed_size, hidden_size, word_vocab, dropout=0.2, fix_grad_idx=60, schema_aggregation='head+tail'):
        super(GraphInputLayer, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word_vocab = word_vocab
        self.fix_grad_idx = fix_grad_idx
        self.word_embed = nn.Embedding(self.word_vocab, self.embed_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.embed_size, self.hidden_size, cell='lstm', schema_aggregation=schema_aggregation)

    def pad_embedding_grad_zero(self, index=None):
        self.word_embed.weight.grad[0].zero_() # padding symbol is always 0
        if index is not None:
            if not torch.is_tensor(index):
                index = torch.tensor(index, dtype=torch.long, device=self.word_embed.weight.grad.device)
            self.word_embed.weight.grad.index_fill_(0, index, 0.)
        else:
            self.word_embed.weight.grad[self.fix_grad_idx:].zero_()

    def forward(self, batch):
        question, table, column = self.word_embed(batch.questions), self.word_embed(batch.tables), self.word_embed(batch.columns)
        if batch.question_unk_mask is not None:
            question = question.masked_scatter_(batch.question_unk_mask.unsqueeze(-1), batch.question_unk_embeddings[:, :self.embed_size])
        if batch.table_unk_mask is not None:
            table = table.masked_scatter_(batch.table_unk_mask.unsqueeze(-1), batch.table_unk_embeddings[:, :self.embed_size])
        if batch.column_unk_mask is not None:
            column = column.masked_scatter_(batch.column_unk_mask.unsqueeze(-1), batch.column_unk_embeddings[:, :self.embed_size])
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs = self.rnn_layer(input_dict, batch)
        return inputs

class GraphInputLayerPLM(nn.Module):

    def __init__(self, plm='bert-base-uncased', hidden_size=256, dropout=0., subword_aggregation='mean',
            schema_aggregation='head+tail', lazy_load=False):
        super(GraphInputLayerPLM, self).__init__()
        self.plm_model = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join('./pretrained_models', plm))) \
            if lazy_load else AutoModel.from_pretrained(os.path.join('./pretrained_models', plm))
        self.config = self.plm_model.config
        self.subword_aggregation = SubwordAggregation(self.config.hidden_size, subword_aggregation=subword_aggregation)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.config.hidden_size, hidden_size, cell='lstm', schema_aggregation=schema_aggregation)

    def pad_embedding_grad_zero(self, index=None):
        pass

    def forward(self, batch):
        outputs = self.plm_model(**batch.inputs)[0] # final layer hidden states
        question, table, column = self.subword_aggregation(outputs, batch)
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs = self.rnn_layer(input_dict, batch)
        return inputs

class SubwordAggregation(nn.Module):
    """ Map subword or wordpieces into one fixed size vector based on aggregation method
    """
    def __init__(self, hidden_size, subword_aggregation='mean-pooling'):
        super(SubwordAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)

    def forward(self, inputs, batch):
        """ Transform pretrained model outputs into our desired format
        questions: bsize x max_question_len x hidden_size
        tables: bsize x max_table_word_len x hidden_size
        columns: bsize x max_column_word_len x hidden_size
        """
        old_questions, old_tables, old_columns = inputs.masked_select(batch.question_mask_plm.unsqueeze(-1)), \
            inputs.masked_select(batch.table_mask_plm.unsqueeze(-1)), inputs.masked_select(batch.column_mask_plm.unsqueeze(-1))
        questions = old_questions.new_zeros(batch.question_subword_lens.size(0), batch.max_question_subword_len, self.hidden_size)
        questions = questions.masked_scatter_(batch.question_subword_mask.unsqueeze(-1), old_questions)
        tables = old_tables.new_zeros(batch.table_subword_lens.size(0), batch.max_table_subword_len, self.hidden_size)
        tables = tables.masked_scatter_(batch.table_subword_mask.unsqueeze(-1), old_tables)
        columns = old_columns.new_zeros(batch.column_subword_lens.size(0), batch.max_column_subword_len, self.hidden_size)
        columns = columns.masked_scatter_(batch.column_subword_mask.unsqueeze(-1), old_columns)

        questions = self.aggregation(questions, mask=batch.question_subword_mask)
        tables = self.aggregation(tables, mask=batch.table_subword_mask)
        columns = self.aggregation(columns, mask=batch.column_subword_mask)

        new_questions, new_tables, new_columns = questions.new_zeros(len(batch), batch.max_question_len, self.hidden_size),\
            tables.new_zeros(batch.table_word_mask.size(0), batch.max_table_word_len, self.hidden_size), \
                columns.new_zeros(batch.column_word_mask.size(0), batch.max_column_word_len, self.hidden_size)
        new_questions = new_questions.masked_scatter_(batch.question_mask.unsqueeze(-1), questions)
        new_tables = new_tables.masked_scatter_(batch.table_word_mask.unsqueeze(-1), tables)
        new_columns = new_columns.masked_scatter_(batch.column_word_mask.unsqueeze(-1), columns)
        return new_questions, new_tables, new_columns

class InputRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cell='lstm', schema_aggregation='head+tail', share_lstm=False):
        super(InputRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell.upper()
        self.question_lstm = getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_lstm = self.question_lstm if share_lstm else \
            getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_aggregation = schema_aggregation
        if self.schema_aggregation != 'head+tail':
            self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=schema_aggregation)

    def forward(self, input_dict, batch):
        """
            for question sentence, forward into a bidirectional LSTM to get contextual info and sequential dependence
            for schema phrase, extract representation for each phrase by concatenating head+tail vectors,
            batch.question_lens, batch.table_word_lens, batch.column_word_lens are used
        """
        questions, _ = rnn_wrapper(self.question_lstm, input_dict['question'], batch.question_lens, cell=self.cell)
        questions = questions.contiguous().view(-1, self.hidden_size)[lens2mask(batch.question_lens).contiguous().view(-1)]
        table_outputs, table_hiddens = rnn_wrapper(self.schema_lstm, input_dict['table'], batch.table_word_lens, cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            tables = self.aggregation(table_outputs, mask=batch.table_word_mask)
        else:
            table_hiddens = table_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else table_hiddens.transpose(0, 1)
            tables = table_hiddens.contiguous().view(-1, self.hidden_size)
        column_outputs, column_hiddens = rnn_wrapper(self.schema_lstm, input_dict['column'], batch.column_word_lens, cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            columns = self.aggregation(column_outputs, mask=batch.column_word_mask)
        else:
            column_hiddens = column_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else column_hiddens.transpose(0, 1)
            columns = column_hiddens.contiguous().view(-1, self.hidden_size)

        questions = questions.split(batch.question_lens.tolist(), dim=0)
        tables = tables.split(batch.table_lens.tolist(), dim=0)
        columns = columns.split(batch.column_lens.tolist(), dim=0)
        # dgl graph node feats format: q11 q12 ... t11 t12 ... c11 c12 ... q21 q22 ...
        outputs = [th for q_t_c in zip(questions, tables, columns) for th in q_t_c]
        outputs = torch.cat(outputs, dim=0)
        # transformer input format: bsize x max([q1 q2 ... t1 t2 ... c1 c2 ...]) x hidden_size
        # outputs = []
        # for q, t, c in zip(questions, tables, columns):
        #     zero_paddings = q.new_zeros((batch.max_len - q.size(0) - t.size(0) - c.size(0), q.size(1)))
        #     cur_outputs = torch.cat([q, t, c, zero_paddings], dim=0)
        #     outputs.append(cur_outputs)
        # outputs = torch.stack(outputs, dim=0) # bsize x max_len x hidden_size
        return outputs
