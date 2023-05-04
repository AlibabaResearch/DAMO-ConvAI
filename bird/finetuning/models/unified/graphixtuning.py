#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoTokenizer
from .base import PushToHubFriendlyModel
#from ..prompt.modeling_auto import AutoModelForSeq2SeqLM
from .constants import GRAPHIX_RELATIONS
from ..FT.modeling_graphix import AutoModelForSeq2SeqLM
from .rgat_tuning import RGAT_Layer
import torch.nn.functional as F
import pickle
import pdb


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location
        )
        self.config = self.pretrain_model.config
        # import rgat_layer:
        self.rgat_layer = T5LayerRGAT(self.config)

        from ..FT.modeling_t5_graphix import T5ForConditionalGeneration
        if isinstance(self.pretrain_model, (T5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        else:
            raise ValueError("Other models are not supported yet!")

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        # Knowledge prompt.
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans_dec = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        self.dropout = nn.Dropout(args.prefix_tuning.prefix_dropout)

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        if self.args.model.freeze_prefix:
            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.control_trans.parameters():
                param.requires_grad = False
            for param in self.wte_dec.parameters():
                param.requires_grad = False
            for param in self.control_trans_dec.parameters():
                param.requires_grad = False
            for param in self.wte_enc.parameters():
                param.requires_grad = False
            for param in self.control_trans_enc.parameters():
                param.requires_grad = False
        
        # add graph part:
        self.graph_pedia = pickle.load(open(args.graph_path.graph_path, 'rb'))
        self.rel2id, self.id2rel = self.enumerate_relation(GRAPHIX_RELATIONS)
        self.relation_embedding = nn.Embedding(len(self.rel2id), self.config.d_model)
    

    def get_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values = torch.cat([past_key_values, self.knowledge_trans(knowledge.repeat_interleave(sample_size, dim=0))], dim=1)

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        if description is not None:
            temp_control_dec = temp_control_dec + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_dec = torch.cat([past_key_values_dec, self.knowledge_trans_dec(knowledge.repeat_interleave(sample_size, dim=0))], dim=1)

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        if description is not None:
            temp_control_enc = temp_control_enc + description.unsqueeze(1)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_enc = torch.cat([past_key_values_enc, self.knowledge_trans_enc(knowledge)], dim=1)

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result

    def get_description_representation(self, kwargs):
        if self.args.model.use_description and self.args.model.map_description:
            description_input_ids = kwargs.pop("description_input_ids")
            description_attention_mask = kwargs.pop("description_attention_mask")
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                description_outputs = self.pretrain_model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                description_outputs = self.pretrain_model.model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            else:
                raise ValueError()
        else:
            description = None

        return description

    def get_knowledge_representation(self, kwargs):
        if self.args.model.knowledge_usage == 'separate':
            knowledge_input_ids = kwargs.pop("knowledge_input_ids", None)
            knowledge_attention_mask = kwargs.pop("knowledge_attention_mask", None)
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                knowledge_outputs = self.pretrain_model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                knowledge_outputs = self.pretrain_model.model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            else:
                raise ValueError()
        elif self.args.model.knowledge_usage == 'concatenate':
            knowledge = None
        else:
            raise ValueError()

        return knowledge
    
    def enumerate_relation(self, relations):
        word2id = {}
        id2word = {}

        for i, r in enumerate(relations):
            word2id[r] = i
            id2word[i] = r

        return word2id, id2word

    def get_graph(self, kwargs):

        ''' load graph_idx and convert into list '''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx[0]) for idx in graph_idx_batch]
        '''load graph_node_idx and convert into list'''
        graph_node_idx_batch = kwargs.pop('graph_nodes_subwords_idx', None)

        new_graph_batch = [] # list of dicts
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_postprocess(self.graph_pedia[graph_idx], device)
            new_graph_batch.append(new_graph)

        return new_graph_batch

    def graph_factory(self, kwargs):

        '''load and postporcess graphs'''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx) for idx in graph_idx_batch]

        new_graph_batch = []
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_postprocess(self.graph_pedia[graph_idx], device)
            new_graph_batch.append(new_graph)
        

        return new_graph_batch
    
    def graph_postprocess(self, graph: dict, device):
        new_graph = {}
        edges = graph['edges']
        rel_ids = list(map(lambda r: self.rel2id[r[2]], edges))

        new_graph['edges'] = torch.tensor(rel_ids, dtype=torch.long, device=device)
        new_graph['graph'] = graph['graph']
        # new_graph['question_subword_mask'] = torch.tensor(graph['question_subword_mask'], dtype=torch.bool)
        # new_graph['schema_subword_mask'] = torch.tensor(graph['schema_subword_mask'], dtype=torch.bool)

        return new_graph

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        # Encode knowledge.
        knowledge_representation = self.get_knowledge_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, description=description_representation, knowledge=knowledge_representation,
        )
        graph_batch = self.graph_factory(kwargs)
        relation_emb=self.relation_embedding

        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
            graph_batch=graph_batch,
            relation_emb=relation_emb,
            rgat_layer=self.rgat_layer
        ).loss
        return {'loss': loss}

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        # Encode knowledge.
        knowledge_representation = self.get_knowledge_representation(kwargs)

        relation_emb=self.relation_embedding

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams'], description=description_representation, knowledge=knowledge_representation,
        )
        graph_batch = self.graph_factory(kwargs)
        # pdb.set_trace()
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            graph_batch=graph_batch,
            rgat_layer=self.rgat_layer,
            relation_emb=relation_emb,
            **kwargs,
        )

        return generated_ids

class T5LayerRGAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rgat = RGAT_Layer(config.d_model, config.d_model, num_heads=1, feat_drop=0.2)
        self.filter = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dropout_gnn = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states, graph_batch, relation_emb):
        hs_norm = self.layer_norm(hidden_states)
        graph_rep = self.graph_caption(hs_norm, graph_batch, relation_emb)

        output = F.elu(graph_rep)
        output = self.dropout_gnn(output)
        # pdb.set_trace()
        # output = self.filter(output)
        output = output.view_as(hidden_states)

        layer_output = hidden_states + self.dropout(output)
        # pdb.set_trace()
        return layer_output
    
    def graph_caption(self, hidden_states, graph_batch, relation_emb):
        '''
        :param hidden_states: [bsz x input_max_length x d_model]
        :return: hidden states with graph caption, while keeping the other reps
        '''
        for i, graph in enumerate(graph_batch):
            hidden_states[i] = self.graph_caption_one(hidden_states[i], graph['graph'], \
                                                      graph['edges'], relation_emb)

        return hidden_states

    def graph_caption_one(self, hidden_state, graph, edges, relation_emb):
        
        num_nodes = graph.number_of_nodes()
        node_feats = hidden_state[:num_nodes]
        edge_feats = relation_emb(edges)
        

        struct_rep, edge_feats = self.rgat(node_feats, edge_feats, graph)
        hidden_state[:num_nodes] = struct_rep
        
        return hidden_state