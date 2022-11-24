from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import copy
from transformers.activations import gelu
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.bert.modeling_bert import *
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.file_utils import add_start_docstrings
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # print('weighted shape',weighted.shape, flush=True)
        if len(weighted.shape) == 2:
            representations = weighted.sum(0).unsqueeze(0)
        else:
            representations = weighted.sum(1).squeeze(1)
        # print('after weighted shape',weighted.shape, flush=True) 

        return representations, scores

class Decoder(GPT2LMHeadModel):
    def forward(
        self,
        input_ids=None,
        latent_proj=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = self.transformer.wte(input_ids)
        # print('input_id',input_ids.size(),'inputs_embs',inputs_embeds.size(),flush=True)
        if latent_proj is not None:
            # print('latent',latent_proj.size(),flush=True)
            # print('before',inputs_embeds.size(),flush=True)
            inputs_embeds = inputs_embeds + latent_proj

        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits, 
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class CVAEModel(nn.Module):
    """CVAE with conditional prior"""

    def __init__(self, config, args):
        super(CVAEModel, self).__init__()
        # gpt2model = GPT2Model(config)
        # self.encoder = gpt2model 
        # self.decoder = Decoder(gpt2model, config)
        self.decoder = Decoder(config)
        self.encoder = self.decoder.transformer

        self.args = args
        self.latent_dim = self.args.z_dim
        
        self.prior_mean = Conv1D(self.latent_dim, config.n_embd)
        self.prior_logvar = Conv1D(self.latent_dim, config.n_embd)
        self.post_mean = Conv1D(self.latent_dim, config.n_embd)
        self.post_logvar = Conv1D(self.latent_dim, config.n_embd)
        self.avg_attn = AverageSelfAttention(config.n_embd)
        self.latent_mlp = nn.Linear(self.latent_dim, config.n_embd, bias=False)

    def initialize(self, path):
        path='gpt2' # add for youqing check.
        self.decoder = Decoder.from_pretrained(path)
        if not self.args.share_params:
            self.encoder = GPT2Model.from_pretrained(path)
        else:
            self.encoder = self.decoder.transformer
    
    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def kl_loss(self, mean1, logvar1, mean2, logvar2): # [batch_size(64), hidden_size(768)]
        # print(mean1.size(),logvar1.size(),mean2.size(),logvar2.size(),flush=True)
        exponential = logvar1 - logvar2 - \
            torch.pow(mean1 - mean2, 2) / logvar2.exp() - \
            torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {"input_ids": input_ids, 'latent_proj': kwargs['latent_proj']}
     
    def forward(self, input_ids, attention_mask, px_tokens=None, px_mask=None, p_tokens=None, p_mask=None):
        # * Get posterior: latent representation p(z|x,p)
        if px_tokens.size() != px_mask.size() or p_tokens.size() != p_mask.size() or input_ids.size() != attention_mask.size():
            raise ValueError('Sizes of input and its mask not match.')
        
        post_out = self.encoder(input_ids=px_tokens, attention_mask=px_mask)
        post_emb, _ = self.avg_attn(post_out[0], attention_mask=px_mask)
        # print('post_emb',post_emb.shape,flush=True) # (64, 768)
        posterior_mean, posterior_logvar = self.post_mean(post_emb), self.post_logvar(post_emb)

        # * Get prior: 
        prior_out = self.encoder(input_ids=p_tokens, attention_mask=p_mask)
        prior_emb, _ = self.avg_attn(prior_out[0], attention_mask=p_mask)
        prior_mean, prior_logvar = self.prior_mean(prior_emb), self.prior_logvar(prior_emb)

        latent_z = self.reparameterize(posterior_mean, posterior_logvar)
        latent_proj = self.args.alpha_z * self.latent_mlp(latent_z)
        assert not torch.isnan(latent_z).any(), 'Training gets NAN z!'

        kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)
        dec_out = self.decoder(input_ids=input_ids, latent_proj=latent_proj)
        outputs = (dec_out.logits, dec_out.past_key_values)

        return outputs + (kl_loss,)
    

class PrevCVAEModel(CVAEModel):
    def __init__(self, config, args):
        super(PrevCVAEModel, self).__init__(config, args)
