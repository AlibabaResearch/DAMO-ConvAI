import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertForNLU(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNLU, self).__init__(config)
        self.class_types = config.class_types
        self.num_class = len(self.class_types)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_class)
        self.intent_loss_fct = CrossEntropyLoss(reduction='mean')
        self.init_weights()
    
    def forward(self,
                input_ids,
                input_mask=None,
                input_type_ids=None,
                matching_label_id=None):
        
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=input_type_ids
        )
        
        pooled_output = outputs[1]
        intent_logits = self.classifier(pooled_output)
        _, preds = torch.max(intent_logits, 1, keepdim=False)
        loss = self.intent_loss_fct(intent_logits, matching_label_id)
        
        return loss, preds
