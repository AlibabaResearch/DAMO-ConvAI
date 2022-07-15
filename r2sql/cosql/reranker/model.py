import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

class ReRanker(nn.Module):
    def __init__(self, args=None, base_model="roberta"):
        super(ReRanker, self).__init__()
        self.args = args
        self.bert_model = RobertaModel.from_pretrained('./local_param/')
        self.cls_model = nn.Sequential(
                        nn.Linear(768, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()
                        )

    def build_optim(self):
        params_cls_trainer = []
        params_bert_trainer = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bert_model' in name:
                    params_bert_trainer.append(param)
                else:
                    params_cls_trainer.append(param)
        self.bert_trainer = torch.optim.Adam(params_bert_trainer, lr=self.args.bert_lr)
        self.cls_trainer = torch.optim.Adam(params_cls_trainer, lr=self.args.cls_lr)


    def forward(self, input_ids, attention_mask):
        # bert model
        x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # get first [CLS] representation
        x = x[:, 0, :]
        # classification layer
        x = self.cls_model(x)
        return x
