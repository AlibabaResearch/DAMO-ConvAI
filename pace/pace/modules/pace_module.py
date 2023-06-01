import torch
import torch.nn as nn
import pytorch_lightning as pl
import pace.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from pace.modules import heads, objectives, pace_utils
from pace.utils import model_state_load
from pace.utils.glossary import acts
import math

class TransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["seq2seq"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            # self.load_from_checkpoint(self.hparams.config["load_path"], map_location="cpu")
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if ckpt["state_dict"]['text_embeddings.position_ids'].shape[1] != self.hparams.config["max_text_len"]:
                state_dict = model_state_load.change_text_maxlen(state_dict, self.hparams.config["max_text_len"])
            if config["loss_names"]["mlm"] > 0 and state_dict["text_embeddings.word_embeddings.weight"].shape[0] < self.hparams.config["vocab_size"]:
                state_dict = model_state_load.resize_token_embedding(state_dict , self.hparams.config["vocab_size"])
            # if self.hparams.config["need_expert_load"] == True:
            #     state_dict = model_state_load.expert_state_load(state_dict)
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        if self.hparams.config["loss_names"]["dst"] > 0:
            self.candidate_value_cache = {}
            self.cross_entropy = nn.CrossEntropyLoss()
            self.dropout = nn.Dropout(config["drop_rate"])
            self.classifier_gate = nn.Linear(hs, 2)
            self.classifier_span = nn.Linear(hs, 3)
            self.classifier_action = nn.Linear(hs, len(acts))
            ## ====init==== ##
            self.classifier_gate.apply(objectives.init_weights)
            self.classifier_span.apply(objectives.init_weights)
            self.classifier_action.apply(objectives.init_weights)

        pace_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def get_extended_attention_mask(self, attention_mask=None):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        return extended_attention_mask

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_segment_ids = batch["text_segment_ids"] if self.hparams.config["use_segment_ids"] else None
        text_embeds = self.text_embeddings(text_ids,token_type_ids=text_segment_ids)
        discard_image = self.hparams.config["discard_image"]

        #TODO 修改mmconv dst相关代码，通过配置discard_image实现图像无关的任务
        # if discard_image:
        #     return self.pure_text_infer(text_ids, text_masks, mask_text=False)
        if imgkey not in batch and image_embeds is None and image_masks is None:
            return self.pure_text_infer(text_ids, text_masks, mask_text=False)
        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]   # 上下无关

            (image_embeds, image_masks, patch_index, image_labels,) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_ids)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        image_patch_len = image_embeds.shape[1]
        co_embeds = torch.cat([image_embeds,text_embeds], dim=1)

        if "attention_masks" in batch:
            max_image_cls_len = self.hparams.config["max_image_len"] + 1#add cls token
            co_masks = batch["attention_masks"][:,max_image_cls_len-image_patch_len:,max_image_cls_len-image_patch_len:]
        else:
            co_masks = torch.cat([image_masks,text_masks], dim=1)

        if discard_image:
            co_embeds = co_embeds[:,image_patch_len:]
            co_masks = co_masks[:, image_patch_len: , image_patch_len:]

        co_masks = self.get_extended_attention_mask(attention_mask=co_masks)       
        x = co_embeds
        it_split = text_embeds.shape[1]
        num_layers = len(self.transformer.blocks)

        #TODO 纯文本的路由
        for i, blk in enumerate(self.transformer.blocks):
            #按任务划分
            if self.hparams.config["loss_names"]["seq2seq"] > 0:
                if i < (num_layers-3):
                    x, _attn = blk(x , mask=co_masks , expert_flag=0 ,it_split=it_split)
                else:
                    x, _attn = blk(x , mask=co_masks , expert_flag=4 ,it_split=it_split)
            else:
                if i < (num_layers-3):
                    x, _attn = blk(x, mask=co_masks, expert_flag=3, it_split=it_split)
                else:
                    x, _attn = blk(x, mask=co_masks, expert_flag=2, it_split=None)

        x = self.transformer.norm(x)
        image_feats, text_feats = (
            x[:, : -text_embeds.shape[1]],
            x[:, -text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(text_feats)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": text_feats[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def pure_text_infer(self, text_ids, text_masks= None, mask_text=False):
        text_embeds = self.text_embeddings(text_ids)
        if text_masks == None:
            text_masks = torch.ones_like(text_ids).to(self.device)
        text_masks = self.get_extended_attention_mask(text_masks)
        text_embeds = (text_embeds + self.token_type_embeddings(torch.zeros_like(text_ids)))
        it_split = text_embeds.shape[1]
        x = text_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=text_masks, expert_flag=0, it_split=it_split , )

        x = self.transformer.norm(x)
        cls_feats = self.pooler(x)
        ret = {
            "text_feats": x,
            "cls_feats": cls_feats,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))
        
        # MMConv DST
        if "dst" in self.current_tasks:
            objectives.set_slot_tokens(self)
            ret.update(objectives.compute_dst(self, batch))
        
        if "intent" in self.current_tasks:
            ret.update(objectives.compute_itm_intent(self, batch))
        
        # Text generation
        if "seq2seq" in self.current_tasks:
            ret.update(objectives.compute_seq2seq(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        pace_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        pace_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        pace_utils.set_task(self)
        output = self(batch)
        ret = dict()
        if self.hparams.config["loss_names"]["intent"] > 0:
            ret["intent_logits"] = output["intent_logits"]
            ret["intent_labels"] = output["intent_labels"]
        return ret

    def validation_epoch_end(self, outs):
        if self.hparams.config["loss_names"]["intent"] > 0:
            objectives.intent_test_wrapup(outs)
        pace_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        pace_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["intent"] > 0:
            ret["intent_logits"] = output["intent_logits"]
            ret["intent_labels"] = output["intent_labels"]
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["intent"] > 0:
            objectives.intent_test_wrapup(outs)
        if self.hparams.config["loss_names"]["seq2seq"] > 0:
            objectives.generation_test_wrapup(self)
        pace_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return pace_utils.set_schedule(self)


'''
    decode part,
    adapt some code from unilm
'''
class TransformerSSDecode(TransformerSS):
    def __init__(self,config , mask_word_id = 103 , eos_id=102 , search_beam_size=10 , ngram_size=3 , 
                    min_len=0 , length_penalty=1.0 , forbid_duplicate_ngrams=False, forbid_ignore_set=None):
        self.mask_word_id = mask_word_id
        self.search_beam_size = search_beam_size
        self.eos_id = eos_id #[SEP]
        self.ngram_size = ngram_size
        self.min_len = min_len
        self.length_penalty = length_penalty
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set=forbid_ignore_set
        super().__init__(config)

    def encode(self, hidden_states , masks , history_states=None , prev_encoded_layers=None , discard_image=True , text_len=0):
        encoded_layers = []
        num_layers = len(self.transformer.blocks)
        for i, blk in enumerate(self.transformer.blocks):
            if discard_image:
                if i < (num_layers-3):
                    hidden_states, _attn = blk(hidden_states, mask=masks, expert_flag=0, it_split=text_len , history_states=history_states)
                else:
                    hidden_states, _attn = blk(hidden_states, mask=masks, expert_flag=4, it_split=text_len , history_states=history_states)
            else:
                if i < (num_layers-3):
                    hidden_states, _attn = blk(hidden_states, mask=masks, expert_flag=3, it_split=text_len , history_states=history_states)
                else:
                    hidden_states, _attn = blk(hidden_states, mask=masks, expert_flag=4, it_split=text_len , history_states=history_states)
            encoded_layers.append(hidden_states)
            
            if prev_encoded_layers is not None:
                history_states = prev_encoded_layers[i]
        hidden_states = self.transformer.norm(hidden_states)
        return hidden_states , encoded_layers

    def forward(self , batch , image_token_type_idx=1 , imgkey="image" , decode_prompt=None):
        if self.search_beam_size > 1:
            return self.beam_search(batch , image_token_type_idx , imgkey, decode_prompt)
        device = self.device.type
        text_ids = batch["text_ids"].to(device) #if "text_ids_mlm" not in batch else batch["text_ids_mlm"].to(device)
        text_masks = batch["text_masks"].to(device)
        position_ids = batch["position_ids"].to(device)
        use_segment_ids = self.hparams.config["use_segment_ids"]
        segment_ids = batch["text_segment_ids"].to(device) if use_segment_ids else None
        # text_embeds = self.text_embeddings(text_ids , token_type_ids=segment_ids)

        img = batch[imgkey][0].to(device)
        (
            image_embeds,
            image_masks,
            patch_index,
            image_labels,
        ) = self.transformer.visual_embed(
            img,
            max_image_len=self.hparams.config["max_image_len"],
            mask_it=False,
        )

        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )
        
        image_patch_len = image_embeds.shape[1]
        max_image_cls_len = self.hparams.config['max_image_len'] + 1
        #如果包含attention_mask , 那么使用attention_mask
        if "attention_masks" in batch:
            co_masks = batch["attention_masks"][:,max_image_cls_len-image_patch_len:,max_image_cls_len-image_patch_len:]
        else:
            co_masks = torch.cat([image_masks, text_masks], dim=1)

        co_masks = self.get_extended_attention_mask(attention_mask=co_masks).to(device)
        
        if decode_prompt != None:
            prompt = torch.tensor(decode_prompt).repeat(text_ids.shape[0], 1).to(device)
            text_ids = torch.cat((text_ids , prompt), dim=1)
        output_ids = []
        batch_size = text_ids.shape[0]
        input_length = text_ids.shape[1]

        next_pos = input_length
        max_length = self.hparams.config["max_text_len"]
        if self.hparams.config["max_pred_len"] >0 :
            max_length = min(max_length,input_length + self.hparams.config["max_pred_len"])
        
        curr_ids = text_ids
        cur_ids = text_ids
        mask_ids = text_ids.new(batch_size,1).fill_(self.mask_word_id)

        discard_image = self.hparams.config['discard_image']
        prev_embeddings = None
        prev_encoded_layers = None
        while next_pos < max_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((cur_ids , mask_ids) , dim=1)
            text_len = x_input_ids.shape[1]
            curr_text_embeds = self.text_embeddings(input_ids=x_input_ids , position_ids=position_ids[: ,:x_input_ids.shape[1]] , 
                token_type_ids=segment_ids[:,:x_input_ids.shape[1]] if use_segment_ids else None) + self.token_type_embeddings(torch.zeros_like(x_input_ids))

            if prev_embeddings is None:
                curr_embeds = torch.cat([image_embeds,curr_text_embeds], dim=1)
                curr_attention_mask = co_masks[:, :,
                                            :image_patch_len+next_pos+1,:image_patch_len+next_pos+1]
            else:
                curr_embeds = curr_text_embeds[:,-2:]
                curr_attention_mask = co_masks[:, :,
                                                image_patch_len+start_pos:image_patch_len+next_pos+1,:image_patch_len+next_pos+1]             
            if discard_image:
                if prev_embeddings is None:
                    curr_embeds = curr_embeds[:,image_patch_len:]
                    curr_attention_mask = curr_attention_mask[:,:, 
                                            image_patch_len: , image_patch_len:]
                else:
                    curr_attention_mask = curr_attention_mask[:,:,
                                            :,image_patch_len:]
            new_hidden_states , new_encoded_layers = self.encode(curr_embeds , curr_attention_mask , history_states=prev_embeddings , prev_encoded_layers=prev_encoded_layers , discard_image=discard_image , text_len=text_len)
            last_feats = new_hidden_states[:,-1:, :]
            prediction_scores = self.mlm_score(last_feats)
            _ , max_ids = torch.max(prediction_scores ,dim=-1)
            output_ids.append(max_ids)

            if prev_embeddings is None:
                prev_embeddings = curr_embeds[:, :-1 ,:]
            else:
                prev_embeddings = torch.cat(
                    (prev_embeddings , curr_embeds[:, :-1 ,:]) , dim=1)
                
            if prev_encoded_layers is None:
                prev_encoded_layers = [layer[:, :-1 , :] for layer in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((xx[0] , xx[1][:, :-1 ,:]) , dim=1)
                                        for xx in zip(prev_encoded_layers , new_encoded_layers)]
            curr_ids = max_ids             
            cur_ids = torch.cat((cur_ids , max_ids), dim=1)
            next_pos += 1

        ret = {"pred_seq":torch.cat(output_ids , dim=1)}
        return ret

    def beam_search(self, batch , image_token_type_idx=1 , imgkey="image" , decode_prompt=None):
        device = self.device.type
        text_ids = batch["text_ids"].to(device)
        text_masks = batch["text_masks"].to(device)
        position_ids = batch["position_ids"].to(device)
        use_segment_ids = self.hparams.config["use_segment_ids"]
        segment_ids = batch["text_segment_ids"].to(device) if use_segment_ids else None

        img = batch[imgkey][0].to(device)        
        (
            image_embeds,
            image_masks,
            patch_index,
            image_labels,
        ) = self.transformer.visual_embed(
            img,
            max_image_len=self.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks , image_token_type_idx))
        max_image_cls_len = self.hparams.config['max_image_len'] + 1
        image_patch_len = image_embeds.shape[1]
        #如果包含attention_mask , 那么使用attention_mask
        if "attention_masks" in batch:
            co_masks = batch["attention_masks"][:,max_image_cls_len-image_patch_len:,max_image_cls_len-image_patch_len:]
        else:
            co_masks = torch.cat([image_masks, text_masks], dim=1)

        co_masks = self.get_extended_attention_mask(attention_mask=co_masks).to(device)

        if decode_prompt != None:
            prompt = torch.tensor(decode_prompt).repeat(text_ids.shape[0], 1).to(device)
            text_ids = torch.cat((text_ids , prompt), dim=1)

        input_shape = list(text_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = text_ids
        cur_ids = text_ids
        mask_ids = text_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        K = self.search_beam_size
        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None
        is_first = True
        max_length = self.hparams.config["max_text_len"]
        if self.hparams.config["max_pred_len"] >0 :
            max_length = min(max_length,input_length + self.hparams.config["max_pred_len"])
        
        discard_image = self.hparams.config["discard_image"]
        while next_pos < max_length:
            curr_length = list(curr_ids.size())[1]

            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((cur_ids, mask_ids), dim=1)
            text_len = x_input_ids.shape[1]
            curr_text_embeds = self.text_embeddings(input_ids=x_input_ids , position_ids=position_ids[: ,:x_input_ids.shape[1]] , 
                token_type_ids=segment_ids[:,:x_input_ids.shape[1]] if use_segment_ids else None) + self.token_type_embeddings(torch.zeros_like(x_input_ids))

            if prev_embedding is None:
                curr_embeds = torch.cat([image_embeds,curr_text_embeds], dim=1)
                curr_attention_mask = co_masks[:, :,
                                                :image_patch_len+next_pos+1,:image_patch_len+next_pos+1]
            else:
                curr_embeds = curr_text_embeds[:,-2:]
                curr_attention_mask = co_masks[:, :,
                                                image_patch_len+start_pos:image_patch_len+next_pos+1,:image_patch_len+next_pos+1]
            if discard_image:
                if prev_embedding is None:
                    curr_embeds = curr_embeds[:, image_patch_len:]
                    curr_attention_mask = curr_attention_mask[:, :, 
                                                image_patch_len:, image_patch_len:]
                else:
                    curr_attention_mask = curr_attention_mask[:, :,
                                                :,image_patch_len:]

            new_hidden_states , new_encoded_layers = self.encode(curr_embeds , curr_attention_mask , history_states=prev_embedding , prev_encoded_layers=prev_encoded_layers , discard_image=discard_image , text_len=text_len)
            last_hidden = new_hidden_states[:, -1:, :]
            prediction_scores = self.mlm_score(last_hidden)
            log_scores = nn.functional.log_softmax(
                prediction_scores, dim=-1)
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K).long()
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)
            if prev_embedding is None:
                prev_embedding = first_expand(curr_embeds[:, :-1, :])
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, curr_embeds[:, :-1, :]), dim=1)
                prev_embedding = select_beam_items(
                    prev_embedding, back_ptrs)

            if prev_encoded_layers is None:
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]
            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])
            if is_first:
                co_masks = first_expand(co_masks)
                mask_ids = first_expand(mask_ids)
                cur_ids = first_expand(cur_ids)
                image_embeds = first_expand(image_embeds)
                position_ids = first_expand(position_ids)
                cur_ids = torch.cat((cur_ids , curr_ids), dim=1)
                if use_segment_ids: segment_ids = first_expand(segment_ids)
            else :
                cur_ids = select_beam_items(cur_ids , back_ptrs)
                cur_ids = torch.cat((cur_ids , curr_ids), dim=1)
                # segment_ids = select_beam_items(segment_ids , back_ptrs)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1
            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims
            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, max_length, padding_value=0).to(device)

        return traces