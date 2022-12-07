import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel


class BertForMatching(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMatching, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.matching_loss_fct = BCEWithLogitsLoss()
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
        hid = self.classifier(pooled_output)
        matching_logits = torch.squeeze(hid, dim=1)
        loss = self.matching_loss_fct(matching_logits, matching_label_id.float())
        
        return loss, matching_logits


