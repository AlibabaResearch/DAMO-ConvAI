import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, BertModel, WavLMModel, RobertaConfig
from wavlm import WavLMForMultiturn, WavLMMAMHead, WavLMForMAM, WavLMForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaLayer, RobertaModel

class SpaceATModel(nn.Module):
    def __init__(self, no_audio, model_dir):
        super(SpaceATModel, self).__init__()
        self.hidden_size = 768
        self.no_audio = no_audio
        # self.audio_encoder = WavLMForMAM.from_pretrained('microsoft/wavlm-base-plus')
        self.audio_encoder = WavLMForMultiturn.from_pretrained(model_dir+'/wavlm-base-plus')
        self.text_encoder = BertModel.from_pretrained(model_dir+'/space3')
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        self.fused_encoder = RobertaLayer(self.text_encoder.config)

    def forward(self, audio_input, text_input, audio_attention_mask=None, text_attention_mask=None, role_token_id=None, turn_id=None):
        if self.no_audio:
            text_features = self.text_encoder(text_input, text_attention_mask, token_type_ids=role_token_id)[0]
            return text_features, text_features[:, 0]
            # audio_input = torch.cat((audio_input[0], audio_input[1]), dim=1)
            # audio_attention_mask = torch.cat((audio_attention_mask[0], audio_attention_mask[1]), dim=1)
        else:
            audio_features, audio_mask = self.audio_encoder(audio_input, audio_attention_mask)
        text_features = self.text_encoder(text_input, text_attention_mask, token_type_ids=role_token_id)[0]
        bs, text_len = text_features.shape[:2]
        token_type_ids = torch.zeros([bs, text_len + audio_features.shape[1]], dtype=torch.long).to(text_input.device)
        token_type_ids[:, text_len:] = 1
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.token_type_embeddings(token_type_ids)
        fused_attention_mask = torch.cat([text_attention_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        fused_features = self.fused_encoder(fused_input, fused_attention_mask)[0]
        return fused_features, fused_features[:, 0]


class DSTModel(nn.Module):
    def __init__(self, no_audio, model_dir, slot_list, class_types, class_labels, dst_dropout_rate, dst_heads_dropout_rate):
        super().__init__()

        self.model = SpaceATModel(no_audio, model_dir) 
        self.slot_list = slot_list
        self.class_types = class_types
        self.class_labels = class_labels
        self.token_loss_for_nonpointable = False
        self.refer_loss_for_nonpointable = False
        self.class_aux_feats_inform = False
        self.class_aux_feats_ds = False
        self.class_loss_ratio = 0.8
        hidden_size = 768

        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        self.dropout = nn.Dropout(dst_dropout_rate)
        self.dropout_heads = nn.Dropout(dst_heads_dropout_rate)

        if self.class_aux_feats_inform:
            self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
        if self.class_aux_feats_ds:
            self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

        aux_dims = len(self.slot_list) * (
                self.class_aux_feats_inform + self.class_aux_feats_ds)  # second term is 0, 1 or 2
        
        for slot in self.slot_list:
            self.add_module("class_" + slot, nn.Linear(hidden_size + aux_dims, self.class_labels))
            self.add_module("token_" + slot, nn.Linear(hidden_size, 2))
            self.add_module("refer_" + slot, nn.Linear(hidden_size + aux_dims, len(self.slot_list) + 1))

        # Head for aux task
        # self.apply(self._init_weights)

    def forward(self, text_input, text_mask, role_token_id, audio_input, audio_mask, turn_id=None,
                start_pos=None, end_pos=None,
                inform_slot_id=None, refer_id=None,
                diag_state=None, class_label_id=None,
                aux_task_def=None):


        sequence_output, pooled_output = self.model(audio_input, text_input, audio_mask, text_mask, role_token_id, turn_id) # last hidden state (b, l, 768)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()  # (b, len(class_types))
        if diag_state is not None:
            diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)  # (b, len(class_types)) 表示slot是否有状态

        total_loss = 0
        per_slot_per_example_loss = {}
        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_refer_logits = {}
        for slot in self.slot_list:
            if self.class_aux_feats_inform and self.class_aux_feats_ds:
                pooled_output_aux = torch.cat(
                    (pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)  # (b, 768+30+30)
            elif self.class_aux_feats_inform:
                pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)  # (b, 768+30)
            elif self.class_aux_feats_ds:
                pooled_output_aux = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)  # (b, 768+30)
            else:
                pooled_output_aux = pooled_output
            # print(pooled_output_aux.shape)
            class_logits = self.dropout_heads(getattr(self, 'class_' + slot)(pooled_output_aux))  # (b, 30)

            token_logits = self.dropout_heads(getattr(self, 'token_' + slot)(sequence_output))  # (b, l, 2)
            start_logits, end_logits = token_logits.split(1, dim=-1)  # (b, l, 1) (b, l, 1)
            # print(start_logits.shape)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            refer_logits = self.dropout_heads(getattr(self, 'refer_' + slot)(pooled_output_aux))  # (b, 31)

            per_slot_class_logits[slot] = class_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits
            per_slot_refer_logits[slot] = refer_logits

            # If there are no labels, don't compute loss
            if class_label_id is not None and start_pos is not None and end_pos is not None and refer_id is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_pos[slot].size()) > 1:
                    start_pos[slot] = start_pos[slot].squeeze(-1)
                if len(end_pos[slot].size()) > 1:
                    end_pos[slot] = end_pos[slot].squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                
                ignored_index = start_logits.size(1)  # l
                start_pos[slot].clamp_(0, ignored_index)
                end_pos[slot].clamp_(0, ignored_index)

                class_loss_fct = CrossEntropyLoss(reduction='none')
                token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)
                refer_loss_fct = CrossEntropyLoss(reduction='none')

                start_loss = token_loss_fct(start_logits, start_pos[slot])
                end_loss = token_loss_fct(end_logits, end_pos[slot])
                token_loss = (start_loss + end_loss) / 2.0

                token_is_pointable = (start_pos[slot] > 0).float()
                if not self.token_loss_for_nonpointable:
                    token_loss *= token_is_pointable

                refer_loss = refer_loss_fct(refer_logits, refer_id[slot])
                # print(class_label_id)
                token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
                if not self.refer_loss_for_nonpointable:
                    refer_loss *= token_is_referrable
                
                class_loss = class_loss_fct(class_logits, class_label_id[slot])

                if self.refer_index > -1:
                    per_example_loss = self.class_loss_ratio * class_loss + (
                            (1 - self.class_loss_ratio) / 2) * token_loss + (
                                               (1 - self.class_loss_ratio) / 2) * refer_loss
                else:
                    per_example_loss = self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss

                total_loss += per_example_loss.sum()
                per_slot_per_example_loss[slot] = per_example_loss

        outputs = (total_loss,) + (per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, 
                  per_slot_end_logits, per_slot_refer_logits,)

        return outputs


