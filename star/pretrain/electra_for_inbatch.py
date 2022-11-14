from transformers.models.electra.configuration_electra import ElectraConfig
import torch
from transformers.models.electra.modeling_electra import ElectraModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraDiscriminatorPredictions
from pooler import *
from torch.nn.functional import gelu
from torch import nn
import os, math
from _utils.model_utils import PoolingFunction,lens2mask,lens2mask2
from _utils.model_utils import PoolingFunction

COLUMN_SQL_LABEL_COUNT = 383

class MLPLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class SubwordAggregation(nn.Module):
    """ Map subword or wordpieces into one fixed size vector based on aggregation method
    """
    def __init__(self, hidden_size, subword_aggregation='mean-pooling'):
        super(SubwordAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)

    def forward(self, inputs, column_mask_plm, column_word_len_mask, max_column_subword_len):
        """ Transform pretrained model outputs into our desired format
        questions: bsize x max_question_len x hidden_size
        tables: bsize x max_table_word_len x hidden_size
        columns: bsize x max_column_word_len x hidden_size
        """
        old_columns = inputs.masked_select(column_mask_plm.unsqueeze(-1))
        
        columns = old_columns.new_zeros(column_word_len_mask.size(0), max_column_subword_len, self.hidden_size)
        columns = columns.masked_scatter_(column_word_len_mask.unsqueeze(-1), old_columns)

        columns = self.aggregation(columns, mask=column_word_len_mask)
        return columns


class Similarity(nn.Module):
  def __init__(self, temp):
    super().__init__()
    self.temp = temp
    self.cos = nn.CosineSimilarity(dim=-1)

  def forward(self, x, y):
    return self.cos(x, y) / self.temp

class Similarity(nn.Module):
  def __init__(self):
    super().__init__()
    self.cos = nn.CosineSimilarity(dim=-1)

  def forward(self, x, y):
    return self.cos(x,y)

class ElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        

        self.dense_col_sss = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm_col_sss = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder_col_sss = nn.Linear(config.hidden_size, COLUMN_SQL_LABEL_COUNT, bias=False)
        self.bias_col_sss = nn.Parameter(torch.zeros(COLUMN_SQL_LABEL_COUNT))

        # self.dense_context_sss = nn.Linear(config.hidden_size, config.hidden_size)
        # self.layer_norm_context_sss = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # self.decoder_context_sss = nn.Linear(config.hidden_size, 1, bias=False)
        # self.bias_context_sss = nn.Parameter(torch.zeros(1))
        self.mlp_sss = MLPLayer(config.hidden_size)
        # self.question_sss = PoolingFunction(self.hidden_size, self.hidden_size, method='mean-pooling')
        # self.question_sss = PoolingFunction(self.hidden_size, self.hidden_size, method='max-pooling')
        self.question_sss = PoolingFunction(config.hidden_size, config.hidden_size, method='attentive-pooling')
        self.column_sss = SubwordAggregation(config.hidden_size, subword_aggregation='attentive-pooling')
        
        self.init_weights()

        self.sim = Similarity()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        question_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        column_mask_plm=None,
        column_word_len=None,
        column_word_num=None,
    ):

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        
        context_outputs = self.mlp_sss(discriminator_sequence_output)
        context_logits = self.question_sss(context_outputs,attention_mask)
        sim_logits = self.sim(context_logits.unsqueeze(1), context_logits.unsqueeze(0))

        column_word_num = lens2mask2(column_word_num,column_word_len.size(1))
        column_word_len = torch.masked_select(column_word_len, column_word_num)
        column_word_len_mask = lens2mask(column_word_len)
        max_column_subword_len = column_word_len_mask.size(1)
        column = self.column_sss(discriminator_sequence_output, column_mask_plm, column_word_len_mask, max_column_subword_len)
        col_prediction_scores = self.dense_col_sss(column)
        col_prediction_scores = gelu(col_prediction_scores)
        col_prediction_scores = self.layer_norm_col_sss(col_prediction_scores)
        col_logits = self.decoder_col_sss(col_prediction_scores) + self.bias_col_sss
        
        context_outputs = self.mlp_sss(discriminator_sequence_output)
        context_logits = self.question_sss(context_outputs,attention_mask)
        
        return logits,  col_logits, context_logits