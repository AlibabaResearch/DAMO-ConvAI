import torch
import bisect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForNextSentencePrediction, AutoModel
from transformers.models.bert.modeling_bert import *

class MarginRankingLoss():
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, p_scores, n_scores):
        scores = self.margin - (p_scores - n_scores)
        scores = scores.clamp(min=0)

        return scores.mean()


class SegModel(nn.Module):
    def __init__(self, model_path='', margin=1, train_split=5, window_size=5):
        super(SegModel, self).__init__()
        self.margin = margin
        self.train_split = train_split
        self.window_size = window_size
        self.topic_model = AutoModel.from_pretrained(model_path+"princeton-nlp/sup-simcse-bert-base-uncased")
        self.coheren_model = BertForNextSentencePrediction.from_pretrained(model_path+"bert-base-uncased", num_labels=2,
                                                                   output_attentions=False,
                                                                   output_hidden_states=True)

        self.topic_loss = nn.CrossEntropyLoss()
        self.score_loss = MarginRankingLoss(self.margin)

    def forward(self, input_data, window_size=None):
        device, topic_loss = input_data['coheren_inputs'].device, torch.tensor(0)
        topic_context_count, topic_pos_count, topic_neg_count = 0, 0, 0
        topic_context_mean, topic_pos_mean, topic_neg_mean = [], [], []
        
        coheren_pos_scores, coheren_pos_feature = self.coheren_model(input_data['coheren_inputs'][:, 0, :],
                                                                    attention_mask=input_data['coheren_mask'][:, 0, :],
                                                                    token_type_ids=input_data['coheren_type'][:, 0, :])
        coheren_neg_scores, coheren_neg_feature = self.coheren_model(input_data['coheren_inputs'][:, 1, :],
                                                                    attention_mask=input_data['coheren_mask'][:, 1, :],
                                                                    token_type_ids=input_data['coheren_type'][:, 1, :])

        batch_size = len(input_data['topic_context_num'])
        topic_context = self.topic_model(input_data['topic_context'], input_data['topic_context_mask'])[1]
        topic_pos = self.topic_model(input_data['topic_pos'], input_data['topic_pos_mask'])[1]
        topic_neg = self.topic_model(input_data['topic_neg'], input_data['topic_neg_mask'])[1]
    
        topic_loss = self.topic_train(input_data, window_size)

        for i, j, z in zip(input_data['topic_context_num'], input_data['topic_pos_num'], input_data['topic_neg_num']):
            topic_context_mean.append(torch.mean(topic_context[topic_context_count:topic_context_count + i], dim=0))
            topic_pos_mean.append(torch.mean(topic_pos[topic_pos_count:topic_pos_count + j], dim=0))
            topic_neg_mean.append(torch.mean(topic_neg[topic_neg_count:topic_neg_count + z], dim=0))
            topic_context_count, topic_pos_count, topic_neg_count = topic_context_count + i, topic_pos_count + j, topic_neg_count + z

        assert len(topic_context_mean) == len(topic_pos_mean) == len(topic_neg_mean) == batch_size

        topic_context_mean, topic_pos_mean = pad_sequence(topic_context_mean, batch_first=True), pad_sequence(topic_pos_mean, batch_first=True)
        topic_neg_mean = pad_sequence(topic_neg_mean, batch_first=True)

        topic_pos_scores = F.cosine_similarity(topic_context_mean, topic_pos_mean, dim=1,
                                            eps=1e-08).to(device)
        topic_neg_scores = F.cosine_similarity(topic_context_mean, topic_neg_mean, dim=1,
                                            eps=1e-08).to(device)

        pos_scores = coheren_pos_scores[0][:, 0] + topic_pos_scores
        neg_scores = coheren_neg_scores[0][:, 0] + topic_neg_scores

        margin_loss = self.score_loss(pos_scores, neg_scores)

        loss = margin_loss.clone() + topic_loss
        return loss, margin_loss, topic_loss

    def infer(self, coheren_input, coheren_mask, coheren_type_id, topic_input=None, topic_mask=None, topic_num=None):
        device = coheren_input.device
        coheren_scores, coheren_feature = self.coheren_model(coheren_input, coheren_mask, coheren_type_id)

        topic_context = self.topic_model(topic_input[0], topic_mask[0])[1]
        topic_cur = self.topic_model(topic_input[1], topic_mask[1])[1]
        topic_context_count = topic_cur_count = 0
        topic_context_mean, topic_cur_mean = [], []

        for i, j in zip(topic_num[0], topic_num[1]):
            topic_context_mean.append(torch.mean(topic_context[topic_context_count:topic_context_count + i], dim=0))
            topic_cur_mean.append(torch.mean(topic_cur[topic_cur_count:topic_cur_count + j], dim=0))
            topic_context_count, topic_cur_count = topic_context_count + i, topic_cur_count + j
        topic_context_mean, topic_cur_mean = pad_sequence(topic_context_mean, batch_first=True), pad_sequence(topic_cur_mean, batch_first=True)
        topic_scores = F.cosine_similarity(topic_context_mean, topic_cur_mean, dim=1, eps=1e-08).to(device)  
        final_scores = coheren_scores[0][:, 0] + topic_scores

        return torch.sigmoid(final_scores).detach().cpu().numpy().tolist()

    def topic_train(self, input_data, window_size):
        device, batch_size = input_data['coheren_inputs'].device, len(input_data['topic_context_num'])
        topic_all = self.topic_model(input_data['topic_train'], input_data['topic_train_mask'])[1]
        true_segments, segments, neg_utts, seg_num, count, margin_count, topic_loss = [], [], [], [], 0, batch_size, torch.tensor(0).to(device, dtype=torch.float)

        # pseudo-segmentation
        for b in range(batch_size):
            cur_num = input_data['topic_num'][b]
            dial_len, cur_utt = cur_num[0], cur_num[1]
            cur = topic_all[count:count+dial_len]
            assert dial_len > cur_utt

            top_cons, top_curs = [], []
            for i in range(1, dial_len):
                top_con = torch.mean(cur[max(0, i-2): i], dim=0)
                top_cur = torch.mean(cur[i: min(dial_len, i+2)], dim=0)
                top_cons.append(top_con)
                top_curs.append(top_cur)
            
            top_cons, top_curs = pad_sequence(top_cons, batch_first=True), pad_sequence(top_curs, batch_first=True)
            topic_scores = F.cosine_similarity(top_cons, top_curs, dim=1, eps=1e-08).to(device)  
            depth_scores = tet(torch.sigmoid(topic_scores))
            tet_seg = np.argsort(np.array(depth_scores))[-self.train_split:] + 1
            tet_seg = [0] + tet_seg.tolist() + [dial_len]
            tet_seg.sort()

            tet_mid = bisect.bisect(tet_seg, cur_utt)
            tet_mid_seg = (tet_seg[tet_mid-1], tet_seg[tet_mid])
                
            pos_left = max(tet_mid_seg[0], cur_utt - window_size)
            pos_right = min(tet_mid_seg[1], cur_utt + window_size+1)

            neg_left = min(tet_seg[max(0, tet_mid-1)], cur_utt - window_size)
            neg_right = max(tet_seg[tet_mid], cur_utt + window_size+1)

            mid = torch.mean(cur[pos_left:pos_right], dim=0).unsqueeze(0)
            segments.append([[mid], (pos_left, pos_right), count, (neg_left, neg_right)])

            count += dial_len

        # Margin loss
        for b in range(batch_size):
            cur_seg = segments[b]
            mid_left, mid_right = cur_seg[1]
            neg_left, neg_right = cur_seg[3]
            count, cur_num = cur_seg[2], input_data['topic_num'][b]
            dial_len, cur_utt, mid_seg = cur_num[0], cur_num[1], cur_seg[0][0]
            
            neg = torch.cat((topic_all[:count+neg_left], topic_all[count+neg_right:]), dim=0)

            anchor = topic_all[count+cur_utt].unsqueeze(0)
            pos = torch.cat((topic_all[count+mid_left:count+cur_utt], topic_all[count+cur_utt+1:count+mid_right]), dim=0)
            pos_score = F.cosine_similarity(anchor, pos, dim=1)
            
            if pos_score.shape[0] == 0:
                margin_count -= 1
                continue
            
            neg_score = F.cosine_similarity(anchor, neg, dim=1)
            margin_pos = pos_score.unsqueeze(0).repeat(neg_score.shape[0], 1).T.flatten()
            margin_neg = neg_score.repeat(pos_score.shape[0])
            assert margin_pos.shape == margin_neg.shape
            cur_loss = self.score_loss(margin_pos, margin_neg)
            if torch.isnan(cur_loss):
                print('Encounter nan:', pos_score.shape, neg_score.shape)
                margin_count -= 1
                continue
            topic_loss += cur_loss

        topic_loss /= margin_count
        return topic_loss


class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_feature = False,
        **kwargs,
    ):

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), pooled_output

def tet(scores):
	output_scores = []
	for i in range(len(scores)):
		lflag, rflag = scores[i], scores[i]
		if i == 0:
			hl = scores[i]
			for r in range(i+1,len(scores)):
				if rflag <= scores[r]:
					rflag = scores[r]
				else:
					break
		elif i == len(scores)-1:
			hr = scores[i]
			for l in range(i-1, -1, -1):
				if lflag <= scores[l]:
					lflag = scores[l]
				else:
					break
		else:
			for r in range(i+1,len(scores)):
				if rflag <= scores[r]:
					rflag = scores[r]
				else:
					break
			for l in range(i-1, -1, -1):
				if lflag <= scores[l]:
					lflag = scores[l]
				else:
					break
		depth_score = 0.5*(lflag+rflag-2*scores[i])
		output_scores.append(depth_score.cpu().detach())

	return output_scores