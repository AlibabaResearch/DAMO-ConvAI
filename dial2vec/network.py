import torch
import torch.nn as nn

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel
from transformers import AutoModel, AutoConfig

from config import huggingface_mapper


class BertAVG(nn.Module):
    """
    对BERT输出的embedding求masked average
    """
    def __init__(self, eps=1e-12):
        super(BertAVG, self).__init__()
        self.eps = eps

    def forward(self, hidden_states, attention_mask):
        mul_mask = lambda x, m: x * torch.unsqueeze(m, dim=-1)
        reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (torch.sum(m, dim=1, keepdims=True) + self.eps)

        avg_output = reduce_mean(hidden_states, attention_mask)
        return avg_output

    def equal_forward(self, hidden_states, attention_mask):
        mul_mask = hidden_states * attention_mask.unsqueeze(-1)
        avg_output = torch.sum(mul_mask, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + self.eps)
        return avg_output


class Dial2vec(nn.Module):
    """
    Dial2vec模型
    """
    def __init__(self, args):
        super(Dial2vec, self).__init__()
        self.args = args
        self.result = {}
        num_labels, total_steps, self.sep_token_id = args.num_labels, args.total_steps, args.sep_token_id

        if args.backbone.lower() == 'plato':
            self.config = PlatoConfig.from_json_file(self.args.config_file)
            self.bert = PlatoModel(self.config)
        elif args.backbone.lower() in ['bert', 'roberta', 'todbert', 't5', 'blender', 'unsup_simcse', 'sup_simcse']:
            self.config = AutoConfig.from_pretrained(huggingface_mapper[args.backbone.lower()])
            self.bert = AutoModel.from_pretrained(huggingface_mapper[args.backbone.lower()])
            # special cases
            if args.backbone.lower() in ['t5']:
                self.config.hidden_dropout_prob = self.config.dropout_rate
                self.bert = self.bert.encoder
            elif args.backbone.lower() in ['blender']:
                self.config.hidden_dropout_prob = self.config.dropout
                self.bert = self.bert.encoder
        else:
            raise NameError('Unknown backbone model: [%s]' % args.backbone)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.labels_data = None
        self.sample_nums = 10
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = BertAVG(eps=1e-6)
        self.logger = args.logger

    def set_finetune(self):
        """
        设置微调层数
        """
        self.logger.debug("******************")
        name_list = ["11", "10", "9", "8", "7", "6"]
        # name_list = ["11",'10','9']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for s in name_list:
                if s in name:
                    self.logger.debug(name)
                    param.requires_grad = True

    def forward(self, data, strategy='mean_by_role', output_attention=False):
        """
        前向传递过程
        """
        if len(data) == 7:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels = data
        else:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels, guids = data

        input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1])
        attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])

        one_mask = torch.ones_like(role_ids)
        zero_mask = torch.zeros_like(role_ids)
        role_a_mask = torch.where(role_ids == 0, one_mask, zero_mask)
        role_b_mask = torch.where(role_ids == 1, one_mask, zero_mask)

        sep_token_id = self.sep_token_id if self.args.use_sep_token else -1

        a_attention_mask = (attention_mask * role_a_mask)
        b_attention_mask = (attention_mask * role_b_mask)

        self_output, pooled_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids)

        q_self_output = self_output * a_attention_mask.unsqueeze(-1)
        r_self_output = self_output * b_attention_mask.unsqueeze(-1)

        self_output = self_output * attention_mask.unsqueeze(-1)
        w = torch.matmul(q_self_output, r_self_output.transpose(-1, -2))

        if turn_ids is not None:
            view_turn_mask = turn_ids.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)
            view_turn_mask_transpose = view_turn_mask.transpose(2, 1)
            view_range_mask = torch.where(abs(view_turn_mask_transpose - view_turn_mask) <= self.args.max_turn_view_range,
                                          torch.ones_like(view_turn_mask),
                                          torch.zeros_like(view_turn_mask))
            filtered_w = w * view_range_mask

        q_cross_output = torch.matmul(filtered_w.permute(0, 2, 1), q_self_output)
        r_cross_output = torch.matmul(filtered_w, r_self_output)

        q_self_output = self.avg(q_self_output, a_attention_mask)
        q_cross_output = self.avg(q_cross_output, b_attention_mask)
        r_self_output = self.avg(r_self_output, b_attention_mask)
        r_cross_output = self.avg(r_cross_output, a_attention_mask)

        self_output = self.avg(self_output, attention_mask)
        q_self_output = q_self_output.view(-1, self.sample_nums, self.config.hidden_size)
        q_cross_output = q_cross_output.view(-1, self.sample_nums, self.config.hidden_size)
        r_self_output = r_self_output.view(-1, self.sample_nums, self.config.hidden_size)
        r_cross_output = r_cross_output.view(-1, self.sample_nums, self.config.hidden_size)

        self_output = self_output.view(-1, self.sample_nums, self.config.hidden_size)
        pooled_output = pooled_output.view(-1, self.sample_nums, self.config.hidden_size)

        output = self_output[:, 0, :]
        q_output = q_self_output[:, 0, :]
        r_output = r_self_output[:, 0, :]
        q_contrastive_output = q_cross_output[:, 0, :]
        r_contrastive_output = r_cross_output[:, 0, :]

        logit_q = []
        logit_r = []
        for i in range(self.sample_nums):
            cos_q = self.calc_cos(q_self_output[:, i, :], q_cross_output[:, i, :])
            cos_r = self.calc_cos(r_self_output[:, i, :], r_cross_output[:, i, :])
            logit_r.append(cos_r)
            logit_q.append(cos_q)

        logit_r = torch.stack(logit_r, dim=1)
        logit_q = torch.stack(logit_q, dim=1)

        loss_r = self.calc_loss(logit_r, labels)
        loss_q = self.calc_loss(logit_q, labels)

        if strategy not in ['mean', 'mean_by_role']:
            raise ValueError('Unknown strategy: [%s]' % strategy)

        output_dict = {'loss': loss_r + loss_q,
                       'final_feature': output if strategy == 'mean' else q_output + r_output,
                       'q_feature': q_output,
                       'r_feature': r_output,
                       'attention': w}

        return output_dict

    def encoder(self, *x):
        """
        BERT编码过程
        """
        input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids = x     # 每个都是[batch_size * num_turn, hidden_size]
        if self.args.backbone in ['bert', 'roberta', 'todbert', 'unsup_simcse', 'sup_simcse']:
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               output_hidden_states=True,
                               return_dict=True)
        elif self.args.backbone in ['t5', 'blender']:
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True,
                               return_dict=True)
            output['pooler_output'] = output['last_hidden_state']   # Notice: 为了实现便利，此处赋值一个tensor占位，但实际上不影响，因为没用到pooler output进行计算。
        elif self.args.backbone in ['plato']:
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               turn_ids=turn_ids,
                               role_ids=role_ids,
                               return_dict=True)
        else:
            raise ValueError('Unknown backbone name: [%s]' % self.args.backbone)

        all_output = output['hidden_states']
        pooler_output = output['pooler_output']
        return all_output[-1], pooler_output

    def calc_cos(self, x, y):
        """
        计算cosine相似度
        """
        cos = torch.cosine_similarity(x, y, dim=1)
        cos = cos / self.args.temperature   # cos = cos / 2.0
        return cos

    def calc_loss(self, pred, labels):
        """
        计算损失函数
        """
        # pred = pred.float()
        loss = -torch.mean(self.log_softmax(pred) * labels)
        return loss

    def get_result(self):
        return self.result

    def get_labels_data(self):
        return self.labels_data