import torch
from torch import nn
from transformers import LongformerModel

from .auxiliary import (BoxEmbedding, 
                        NoCorefHead, 
                        FashionEncoderHead, 
                        FurnitureEncoderHead, 
                        DisambiguationHead, 
                        IntentHead)
from .auxiliary import (FashionTypeHead,
                        FashionAvaliableSizeHead,
                        FashionBrandHead,
                        FashionColorHead,
                        FashionCustomerReviewHead,
                        FashionPatternHead,
                        FashionPriceHead,
                        FashionSizeHead,
                        FashionSleeveLengthHead)
from .auxiliary import (FurnitureBrandHead,
                        FurnitureColorHead,
                        FurnitureCustomerRatingHead,
                        FurnitureMaterialHead,
                        FurniturePriceHead,
                        FurnitureTypeHead)

from .auxiliary import (DisamAllHead,
                        DisamTypeHead)


class VLBertModel(nn.Module):
    ''' 将整体结构封装为一个模型'''

    def __init__(self, args):
        super(VLBertModel, self).__init__()
        self.encoder = LongformerModel.from_pretrained(args.backbone)

        if args.add_special_tokens:
            self.encoder.resize_token_embeddings(args.len_tokenizer)
            self.encoder.vocab_size = args.len_tokenizer

        self.box_embedding = BoxEmbedding(self.encoder.config.hidden_size)
        self.nocoref_head = NoCorefHead(self.encoder.config.hidden_size)

        self.fashion_enc_head = FashionEncoderHead(self.encoder.config.hidden_size)
        self.furniture_enc_head = FurnitureEncoderHead(self.encoder.config.hidden_size)
        
        self.disambiguation_head = DisambiguationHead(self.encoder.config.hidden_size)

        self.CELoss = nn.CrossEntropyLoss()  # 多分类
        self.BCELoss = nn.BCEWithLogitsLoss()  # 多个2分类，并且不需要经过Sigmod
        

    def evaluate(self, enc_input, enc_attention_mask, boxes, misc, disambiguation_label):
        '''  评测任务2在VLBERT模型上的表现效果'''
        batch_size = len(misc)
        inputs_embeds = self.encoder.embeddings(enc_input)
            
        for b_idx in range(batch_size):  # in a batch
            box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(inputs_embeds.device))  # (num_obj_per_line, d_model)
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
        
        enc_last_state = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=enc_attention_mask
        ).last_hidden_state

        disambiguation_pred = self.disambiguation_head(enc_last_state[:, 1, :]).argmax(dim=1)
        disambiguation_true_items = (disambiguation_label == disambiguation_pred).sum().item()  # averaged over a batch (0~1)
        disambiguation_total_items = batch_size

        n_true_objects, n_pred_objects, n_correct_objects = 0, 0, 0

        for b_idx in range(batch_size):  # in a batch

            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                # hidden_concat: (num_obj, 2*model)
                if obj_idx == 0:
                    hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1))
                else:
                    hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1))], dim=0)

            is_fashion = misc[b_idx][0]['is_fashion']
            coref_label = torch.tensor([misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]).to(inputs_embeds.device)  # (num_obj)  0 or 1
            n_true_objects += coref_label.sum().item()

            if is_fashion:
                coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, price, customer_review = self.fashion_enc_head(hidden_concat)  # (num_obj, num_logits)
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item() # 1. or 0.
            else:
                coref, brand, color, materials, type_, price, customer_review = self.furniture_enc_head(hidden_concat)  # (num_obj, num_logits)
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item() # 1. or 0.

        return n_true_objects, n_pred_objects, n_correct_objects, disambiguation_true_items, disambiguation_total_items, disambiguation_pred.tolist()


    def evaluate_for_disam(self, enc_input, enc_attention_mask, boxes, misc, disambiguation_label):
        '''  评测任务1在VLBERT模型上的表现效果'''

        batch_size = len(misc)
        inputs_embeds = self.encoder.embeddings(enc_input)
        
        for b_idx in range(batch_size):  # in a batch
            box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(inputs_embeds.device))  # (num_obj_per_line, d_model)
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
        
        enc_last_state = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=enc_attention_mask
        ).last_hidden_state

        disambiguation_logits = self.disambiguation_head(enc_last_state[:, 1, :])     
        disambiguation_pred = disambiguation_logits.argmax(dim=1)

        disambiguation_true_items = (disambiguation_label == disambiguation_pred).sum().item()  # averaged over a batch (0~1)
        disambiguation_total_items = batch_size

        n_true_objects, n_pred_objects, n_correct_objects = 0, 0, 0

        for b_idx in range(batch_size):  # in a batch

            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                if obj_idx == 0:
                    hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1)) # hidden_concat: (num_obj, 2*model)
                else:
                    hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1))], dim=0)

            is_fashion = misc[b_idx][0]['is_fashion']
            coref_label = torch.tensor([misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]).to(inputs_embeds.device)  # (num_obj)  0 or 1
            
            if disambiguation_label[b_idx].item() == 1:
                n_true_objects += coref_label.sum().item()

            if disambiguation_pred[b_idx].item() != 1:
                continue

            if is_fashion:
                coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, price, customer_review = self.fashion_enc_head(hidden_concat)  # (num_obj, num_logits)
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item() # 1. or 0.
            else:
                coref, brand, color, materials, type_, price, customer_review = self.furniture_enc_head(hidden_concat)  # (num_obj, num_logits)
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item() # 1. or 0.

        return n_true_objects, n_pred_objects, n_correct_objects, disambiguation_true_items, disambiguation_total_items
