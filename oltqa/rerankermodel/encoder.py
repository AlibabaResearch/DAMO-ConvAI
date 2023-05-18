import torch
from torch.cuda import amp
from transformers import (
    RobertaPreTrainedModel,
    RobertaModel,
    BertModel,
    BertPreTrainedModel,
)


class BertEncoder_For_CrossEncoder(BertPreTrainedModel):
    """
    Encoder for crossencoder using BertModel as a backbone model

    In the case of crossencoders,
     questions and phrases are combined,
     and the scalar value obtained by passing the final cls token through the linear layer
     is used as a score for the similarity of the q-p pair.
    """

    def __init__(self, config):
        super(BertEncoder_For_CrossEncoder, self).__init__(config)

        self.bert = BertModel(config)  # Call BertModel
        self.init_weights()  # initalized Weight
        classifier_dropout = (  # Dropout
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.linear = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with amp.autocast(enabled=True):
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,  # If you want to use Roberta Model, Comment out this code
            )

            pooled_output = outputs[1]  # CLS pooled output
            pooled_output = self.dropout(pooled_output)  # apply dropout
            output = self.linear(pooled_output)  # apply classifier
            return output


class RoBertaEncoder_For_CrossEncoder(RobertaPreTrainedModel):
    """
    Encoder for crossencoder using RoBertaModel as a backbone model

    In the case of crossencoders,
     questions and phrases are combined,
     and the scalar value obtained by passing the final cls token through the linear layer
     is used as a score for the similarity of the q-p pair.
    """

    def __init__(self, config):
        super(RoBertaEncoder_For_CrossEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)  # Call RobertaModel
        self.init_weights()  # initalized Weight
        classifier_dropout = (  # Dropout
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.linear = torch.nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        # token_type_ids=None
    ):
        with amp.autocast(enabled=True):
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids
            )

            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            output = self.linear(pooled_output)
            return output


class BertEncoder_For_BiEncoder(BertPreTrainedModel):
    """
    Encoder for bi-encoder using BertModel as a backbone model

    In the case of a bi-encoder,
     the question and phrase each have a hidden embedding
     for the cls token as the final output.
    """

    def __init__(self, config):
        super(BertEncoder_For_BiEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output
 