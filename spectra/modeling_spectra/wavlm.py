import math
import random
import warnings
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional, Tuple
from transformers.activations import ACT2FN
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import WavLMConfig

_HIDDEN_STATES_START_POSITION = 2
MASK_CONSECUTIVE_MIN = 20
MASK_CONSECUTIVE_MAX = 50
MASK_PROPORTION = 0.15
MASK_BLOCK = 1.2


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->WavLM
class WavLMNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id],
                              stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->WavLM
class WavLMLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id],
                              stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->WavLM
class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id],
                              stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.activation = ACT2FN[config.feat_extract_activation]
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->WavLM
class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.num_conv_pos_embeddings,
                              padding=config.num_conv_pos_embeddings // 2, groups=config.num_conv_pos_embedding_groups)
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.padding = WavLMSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->WavLM
class WavLMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder with Wav2Vec2->WavLM
class WavLMFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [WavLMGroupNormConvLayer(config, layer_id=0)]
            for i in range(1, config.num_feat_extract_layers):
                conv_layers.append(WavLMNoLayerNormConvLayer(config, layer_id=i))
            # conv_layers.append(WavLMGroupNormConvLayer(config, layer_id=config.num_feat_extract_layers - 1))
        elif config.feat_extract_norm == "layer":
            conv_layers = [WavLMLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(f"config.feat_extract_norm {config.feat_extract_norm} has to be one of ['group', 'layer']")
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


class WavLMFeatureExtractor(WavLMFeatureEncoder):
    def __init__(self, config):
        super().__init__(config)
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->WavLM
class WavLMFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class WavLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, num_buckets=320, max_distance=800,
                 has_relative_position_bias=True, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        """Attention layer with relative attention"""
        bsz, tgt_len, _ = hidden_states.size()
        # first pass of attention layer creates position bias
        if position_bias is None:
            position_bias = self.compute_bias(tgt_len, tgt_len, hidden_states.device)
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)
            )
        # Compute relative position bias:
        # 1) get reshape hidden_states
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)
        # 2) project hidden states
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)
        # 3) compute gate for position bias from projected hidden states
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        # 4) apply gate to position bias to compute gated position_bias
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))

        attn_output, attn_weights = self.torch_multi_head_self_attention(
            hidden_states, attention_mask, gated_position_bias, output_attentions
        )

        return attn_output, attn_weights, position_bias

    def torch_multi_head_self_attention(self, hidden_states, attention_mask, gated_position_bias, output_attentions):
        """simple wrapper around torch's multi_head_attention_forward function"""
        # self-attention assumes q = k = v
        query = key = value = hidden_states.transpose(0, 1)
        key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None

        # disable bias and add_zero_attn
        bias_k = bias_v = None
        add_zero_attn = False

        # PyTorch 1.3.0 has F.multi_head_attention_forward defined
        # so no problem with backwards compatibility
        attn_output, attn_weights = nn.functional.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k,
            bias_v,
            add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            output_attentions,
            gated_position_bias,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )

        # [Seq_Len, Batch Size, ...] -> [Batch Size, Seq_Len, ...]
        attn_output = attn_output.transpose(0, 1)

        if attn_weights is not None:
            # IMPORTANT: Attention weights are averaged weights
            # here which should not be the case. This is an open issue
            # on PyTorch: https://github.com/pytorch/pytorch/issues/32590
            attn_weights = attn_weights[:, None].broadcast_to(
                attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
            )

        return attn_output, attn_weights

    def compute_bias(self, query_length, key_length, device):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :].to(device)
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position).to(device)
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        num_buckets = self.num_buckets // 2

        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        return relative_buckets


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->WavLM
class WavLMFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = WavLMAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads,
                                        dropout=config.attention_dropout, num_buckets=config.num_buckets,
                                        max_distance=config.max_bucket_distance,
                                        has_relative_position_bias=has_relative_position_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(hidden_states, attention_mask=attention_mask,
                                                                    position_bias=position_bias,
                                                                    output_attentions=output_attentions, index=index)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = WavLMAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads,
                                        dropout=config.attention_dropout, num_buckets=config.num_buckets,
                                        max_distance=config.max_bucket_distance,
                                        has_relative_position_bias=has_relative_position_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(hidden_states, attention_mask=attention_mask,
                                                                    position_bias=position_bias,
                                                                    output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class WavLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False,
                return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), hidden_states,
                                                                      attention_mask, position_bias)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, position_bias=position_bias,
                                          output_attentions=output_attentions, index=i)
                hidden_states, position_bias = layer_outputs[:2]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class WavLMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([WavLMEncoderLayerStableLayerNorm(config, has_relative_position_bias=(i == 0))
                                     for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False,
                return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), hidden_states,
                                                                      attention_mask, position_bias)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, position_bias=position_bias,
                                          output_attentions=output_attentions, index=i)
                hidden_states, position_bias = layer_outputs[:2]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter with Wav2Vec2->WavLM
class WavLMAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # feature dim might need to be down-projected
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        self.layers = nn.ModuleList(WavLMAdapterLayer(config) for _ in range(config.num_adapter_layers))
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer with Wav2Vec2->WavLM
class WavLMAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


class WavLMMAMHead(nn.Module):
    """WavLM Head for masked audio modeling."""

    def __init__(self, hidden_size, output_dim, hidden_act: str = "gelu", dr=1, layer_norm_eps=1e-5):
        super().__init__()
        self.output_dim = output_dim
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act_fn = ACT2FN[hidden_act]
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, self.output_dim * dr, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_dim * dr))
        self.decoder.bias = self.bias

    def forward(self, features):
        x = self.dense(features)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class WavLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = WavLMConfig
    base_model_prefix = "wavlm"
    main_input_name = "input_values"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, WavLMPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, WavLMFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='trunc') + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask, add_adapter=None):
        non_padded_lengths = torch.sum(attention_mask, dim=-1)
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = torch.clamp_max(output_lengths + 1, feature_vector_length).long()
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros((batch_size, feature_vector_length)).to(attention_mask.device,
                                                                             dtype=attention_mask.dtype)
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return output_lengths.tolist(), attention_mask

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder)):
            module.gradient_checkpointing = value


def select_interval(audio_length, mode="round"):
    mask_consecutive = random.randint(MASK_CONSECUTIVE_MIN, MASK_CONSECUTIVE_MAX)  # mask区间长度
    valid_start_max = max(audio_length - mask_consecutive - 1, 0)  # mask区间的起点
    if mode == "round":
        # 先计算有多少个MASK然后再决定位置
        proportion = round(audio_length * MASK_PROPORTION / mask_consecutive)  # 结合mask长度考虑后，得到mask的概率
        chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]  # 允许多个mask重叠
    else:
        # 不决定MASK位置
        chosen_starts = []
        i = 0
        while i < audio_length - mask_consecutive:
            r = random.random()
            if r < MASK_PROPORTION:
                chosen_starts.append(i)
                i += round(mask_consecutive * MASK_BLOCK)
            i += 1
        chosen_starts = torch.LongTensor(chosen_starts)
    tiled = chosen_starts.expand(mask_consecutive, chosen_starts.size(0)).permute(1, 0)
    offset = torch.arange(mask_consecutive).expand_as(tiled)
    intervals = tiled + offset
    return intervals.view(-1)  # 被mask的所有位置


def create_mam_samples(audio, audio_len):
    # spec_masked：输入 spec_stacked：target
    dtype = audio.dtype
    labels = audio.clone()
    masked = torch.zeros(labels.shape[:2] + (1,), dtype=torch.uint8).to(audio.device)
    for idx in range(labels.shape[0]):
        chosen_intervals = select_interval(audio_len[idx], "mlm")
        dice = np.random.uniform(0, 1, len(chosen_intervals))
        # 以80%的概率替换为0，10%的概率替换为序列中的其他token，10%的概率不做修改。音频的mask会mask一整个token
        zero_intervals = torch.BoolTensor(dice < 0.8)
        zero_intervals = torch.masked_select(chosen_intervals, zero_intervals)
        rand_intervals = torch.BoolTensor((dice >= 0.8) * (dice < 0.9))
        rand_intervals = torch.masked_select(chosen_intervals, rand_intervals)
        if len(zero_intervals) > 0:
            audio[idx, zero_intervals, :] = 0
        masked[idx, chosen_intervals, :] = 1
        if len(rand_intervals) > 0:
            random_intervals = torch.randperm(audio_len[idx])[:len(rand_intervals)]
            audio[idx, rand_intervals, :] = labels[idx, random_intervals, :]
    return audio.to(dtype=dtype), masked.to(dtype=torch.bool), labels.to(dtype=dtype)


class WavLMForMultiTurn(WavLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"conv_layers\.7", r"token_embedding", r"audio_cls", r"audio_sep"]
    _keys_to_ignore_on_load_unexpected = [r"masked_spec_embed"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = WavLMFeatureEncoder(config)
        self.feature_projection = WavLMFeatureProjection(config)
        self.audio_cls = nn.Parameter(torch.randn(config.hidden_size), requires_grad=True)
        self.audio_sep = nn.Parameter(torch.randn(config.hidden_size), requires_grad=True)
        if config.do_stable_layer_norm:
            self.encoder = WavLMEncoderStableLayerNorm(config)
        else:
            self.encoder = WavLMEncoder(config)
        self.adapter = WavLMAdapter(config) if config.add_adapter else None
        self.post_init()

    def forward(self, input_values, attention_mask, bs, perform_mam=False, token_embedding=None):
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        out_len, attention_mask = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )
        mam_labels, masked_indices, masked_indices_for_concat = None, None, None
        if perform_mam:
            extract_features, masked_indices, mam_labels = create_mam_samples(extract_features, out_len)
        hidden_states, extract_features = self.feature_projection(extract_features)
        if input_values.shape[0] / bs == 3:
            audio_len = extract_features.shape[1]
            new_audio_len = audio_len * 2 + 2
            aa, pa, na = torch.split(hidden_states.view(bs, 3, -1, self.config.hidden_size), 1, dim=1)
            am, pm, nm = torch.split(attention_mask.view(bs, 3, -1), 1, dim=1)
            aa, pa, na, am, pm, nm = map(lambda x: x.squeeze(1), [aa, pa, na, am, pm, nm])
            hidden_states = torch.zeros([bs * 2, new_audio_len, self.config.hidden_size],
                                        device=aa.device, dtype=aa.dtype)
            attention_mask = torch.zeros([bs * 2, new_audio_len], device=aa.device, dtype=torch.bool)
            token_type_id = torch.ones([bs * 2, new_audio_len], device=aa.device, dtype=torch.long)
            masked_indices_for_concat = torch.zeros([bs, new_audio_len], device=aa.device, dtype=masked_indices.dtype)
            for i in range(bs):
                laa, lpa, lna = out_len[3 * i: 3 * i + 3]
                pl = laa + lpa + 2
                hidden_states[2 * i, :pl] = torch.cat([self.audio_cls.unsqueeze(0), aa[i][:laa],
                                                       self.audio_sep.unsqueeze(0), pa[i][:lpa]], dim=0)
                attention_mask[2 * i, :pl] = True
                nl = laa + lna + 2
                hidden_states[2 * i + 1, :nl] = torch.cat([self.audio_cls.unsqueeze(0), aa[i][:laa],
                                                           self.audio_sep.unsqueeze(0), na[i][:lna]], dim=0)
                attention_mask[2 * i + 1, :nl] = True
                masked_indices_for_concat[i, :pl] = torch.cat([torch.tensor([0]).bool().to(aa.device),
                                                               masked_indices[3 * i][:laa, 0],
                                                               torch.tensor([0]).bool().to(aa.device),
                                                               masked_indices[3 * i + 1][:lpa, 0]], dim=0)
                token_type_id[2 * i: 2 * i + 2, :laa + 1] = 0
            masked_indices = masked_indices.view(bs, 3, audio_len, 1)[:, :2].contiguous().view(bs * 2, audio_len, 1)
            mam_labels = mam_labels.view(bs, 3, audio_len, -1)[:, :2].contiguous().view(bs * 2, audio_len, -1)
        elif input_values.shape[0] / bs == 2:
            audio_len = hidden_states.shape[1]
            new_len = audio_len * 2 + 2
            bs = hidden_states.shape[0] // 2
            a1, a2 = torch.split(hidden_states.view(bs, 2, -1, self.config.hidden_size), 1, dim=1)
            m1, m2 = torch.split(attention_mask.view(bs, 2, -1), 1, dim=1)
            a1, a2, m1, m2 = map(lambda x: x.squeeze(1), [a1, a2, m1, m2])
            hidden_states = torch.zeros([bs, new_len, self.config.hidden_size], device=a1.device, dtype=a1.dtype)
            attention_mask = torch.zeros([bs, new_len], device=a1.device, dtype=torch.bool)
            token_type_id = torch.ones([bs, new_len], device=a1.device, dtype=torch.long)
            for i in range(bs):
                la1, la2 = out_len[2 * i: 2 * i + 2]
                pl = la1 + la2 + 2
                hidden_states[i, :pl] = torch.cat(
                    [self.audio_cls.unsqueeze(0), a1[i, :la1], self.audio_sep.unsqueeze(0), a2[i, :la2]], dim=0)
                token_type_id[i, :la1 + 1] = 0
                attention_mask[i, :pl] = True
        else:
            if self.config.has_audio_cls:
                bs = hidden_states.shape[0]
                attention_mask = torch.cat([torch.zeros(bs, 1).long().to(input_values.device), attention_mask], dim=1)
                hidden_states = torch.cat([self.audio_cls[None, None, :].repeat(bs, 1, 1), hidden_states], dim=1)
            token_type_id = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=input_values.device)
        if token_embedding is not None:
            hidden_states = hidden_states + token_embedding(token_type_id)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        if perform_mam:
            return hidden_states, attention_mask, mam_labels, (masked_indices_for_concat.unsqueeze(-1), masked_indices)
        return hidden_states, attention_mask
