import copy

import torch
from einops import rearrange, repeat
from torch import einsum, nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen2ForCausalLM
from openomni.constants import IGNORE_INDEX


def exists(val):
    return val is not None

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class CrossAttention(nn.Module):
    def __init__(self, dim=896, dim_head=256, heads=16):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_text = nn.LayerNorm(dim)
        self.norm_image = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
    def forward(self, text_reps, image_reps, text_mask=None, image_mask=None):
        """
        Args:
            text_reps (torch.Tensor): text features
                shape (b, n1, D)
            image_reps (torch.Tensor): image features
                shape (b, n2, D)
            text_mask (torch.BoolTensor): mask for text
                shape (b, n1)
            image_mask (torch.BoolTensor): mask for image
                shape (b, n2)
        """
        text_reps = self.norm_text(text_reps)
        image_reps = self.norm_image(image_reps)
        h = self.heads
        q = self.to_q(image_reps)
        # kv_input = torch.cat((image_reps, text_reps), dim=-2)
        kv_input=text_reps
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange(q, "b n (h d) -> b h n d", h=h), rearrange(k, "b n (h d) -> b h n d", h=h), rearrange(v, "b n (h d) -> b h n d", h=h)
        q = q * self.scale
        
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        
        # mask processing
        if text_mask is not None:
            # mask=torch.cat([image_mask,text_mask], dim=-1)
            mask=text_mask
            mask = rearrange(mask, "b n -> b 1 1 n")  # Expand dims for broadcasting
            sim = sim.masked_fill(mask == False, float('-inf'))

        # sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)

class RLSIAT(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        dim_head=256,
        heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CrossAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, text_reps, image_reps, text_mask=None, image_mask=None):

        b,n,_=text_reps.shape
        for attn, ff in self.layers:
            image_reps = attn(text_reps, image_reps, text_mask=text_mask,image_mask=image_mask) + image_reps
            image_reps = ff(image_reps) + image_reps
        return self.norm(image_reps)

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def _uniform_assignment(src_lens, tgt_lens):
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)
    ratio = tgt_lens / src_lens
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = Gate(input_dim, num_experts)
        self.num_experts=num_experts

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        x = x.view(-1, dim)  # Flatten the input for gating network
        
        # 通过门控网络计算每个expert的选择
        gating_scores = self.gate(x)  # (batch_size * seq_length, num_experts)
        
        # 将 gating_scores 反向重塑为原来的形状
        gating_scores = gating_scores.view(batch_size, seq_length, self.num_experts)
        
        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(x).view(batch_size, seq_length, -1) for expert in self.experts], dim=1)  # (batch_size, num_experts, seq_length, output_dim)
        # print(expert_outputs.shape)
        expert_outputs = expert_outputs.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_experts, output_dim)

        # 使用 gating_scores 加权专家输出
        outputs = (gating_scores.unsqueeze(-1) * expert_outputs).sum(dim=2)  # (batch_size, seq_length, output_dim)
        
        return outputs

class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache

class SpeechGeneratorCTC(nn.Module):
    def __init__(self, config):
        super().__init__()
        # print(config)
        n_layers, n_dims, n_heads, n_inter_dims = list(map(int, config.ctc_decoder_config[1:-1].split(",")))
        # print(n_layers, n_dims, n_heads, n_inter_dims)
        _config = copy.deepcopy(config)
        _config.hidden_size = n_dims
        _config.num_hidden_layers = n_layers
        _config.num_attention_heads = n_heads
        _config.num_key_value_heads = n_heads
        _config.intermediate_size = n_inter_dims
        _config._attn_implementation = "flash_attention_2"

        self.unit_vocab_size = config.unit_vocab_size

        qwen_tiny_path="Qwen/Qwen2.5-0.5B-Instruct"
        self.llm = Qwen2Encoder(qwen_tiny_path)

        self.n_dims=896 # fix
        self.llm_input_size = self.n_dims
        self.llm_output_size = self.n_dims

        self.upsample_factor = config.ctc_upsample_factor

        # modules = [nn.Linear(config.hidden_size, self.n_dims*4)]
        # modules.append(nn.GELU())
        # modules.append(nn.Linear(self.n_dims*4, self.n_dims*4))

        # self.input_proj = nn.Sequential(*modules)
        self.input_proj=MoE(config.hidden_size, self.n_dims*4, 4)
        # self.upsample_proj=nn.Linear(self.n_dims, self.n_dims*self.upsample_factor)

        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, config.hidden_size)

        self.llm_decoder = nn.Linear(self.llm_output_size, self.unit_vocab_size + 1)

        self.fusion=RLSIAT(self.n_dims)
        

    def upsample(self,reps, tgt_units=None):
        src_lens = torch.LongTensor([len(rep) for rep in reps]).to(reps[0].device)
        up_lens = src_lens * self.upsample_factor
        if tgt_units is not None:
            tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
            # print(tgt_lens/ src_lens)
            up_lens = torch.max(up_lens, tgt_lens)
        
        # max_len=torch.max(up_lens)
        # dummy_rep=torch.full((max_len , reps[0].shape[-1]), 0, dtype=reps[0].dtype, device=reps[0].device)
        reps = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True, padding_value=0)
        padding_mask = lengths_to_padding_mask(up_lens)
        # print(padding_mask.shape)
        mapped_inputs = _uniform_assignment(src_lens, up_lens).masked_fill(
            padding_mask, 0
        )

        copied_reps = torch.gather(
            reps,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), reps.size(-1)
            ),
        )

        copied_reps = copied_reps.masked_fill(padding_mask.unsqueeze(-1), 0)
        # print(reps.shape,padding_mask.shape,copied_reps.shape)
        position_ids = torch.arange(0, max(up_lens)).unsqueeze(0).expand(len(reps), -1).to(device=copied_reps.device)
        return copied_reps, ~padding_mask, position_ids
    
    def forward(self, text_reps, text_labels, speech_tokens, text_tokens=None, embedding=None):
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        tgt_text_reps = []
        for text_rep, text_label in zip(text_reps, text_labels):  
            tgt_text_reps.append(torch.cat([sos_eos_emb.squeeze(dim=0),
            text_rep[text_label != IGNORE_INDEX],
            task_id_emb.squeeze(dim=0)]))
        del text_reps
        text_reps_lens = torch.LongTensor([len(rep)*4 for rep in tgt_text_reps]).to(tgt_text_reps[0].device)
        tgt_text_reps_padding_mask = ~lengths_to_padding_mask(text_reps_lens)
        tgt_text_reps = torch.nn.utils.rnn.pad_sequence(tgt_text_reps, batch_first=True)
        tgt_text_reps = self.input_proj(tgt_text_reps)
        tgt_text_reps = rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
        # text_embeddings = self.input_proj(tgt_text_embeddings)
        text_token_lens = text_tokens.ne(IGNORE_INDEX).long().sum(dim=-1)
        # print(text_token_lens)
        text=torch.nn.utils.rnn.pad_sequence([x[x.ne(IGNORE_INDEX)] for x in text_tokens], batch_first=True, padding_value=0)
        text_embeddings=self.llm.model.model.embed_tokens(text)
        text_embeddings=self.fusion(tgt_text_reps,text_embeddings,tgt_text_reps_padding_mask,None)
        # try:
        tgt_text_embeddings=[text_embedding[:text_token_len] for (text_token_len, text_embedding) in zip(text_token_lens, text_embeddings)]
        new_input_embeds, attention_mask, position_ids = self.upsample(tgt_text_embeddings, speech_tokens)
        outputs=self.llm.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=new_input_embeds,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        del outputs.logits
        ctc_logits = self.llm_decoder(outputs.hidden_states[-1])
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_lens = attention_mask.long().sum(dim=-1)
        ctc_tgt_lens = speech_tokens.ne(IGNORE_INDEX).long().sum(dim=-1)
        ctc_tgt_mask = ~lengths_to_padding_mask(ctc_tgt_lens)
        ctc_tgt_flat = speech_tokens.masked_select(ctc_tgt_mask)
        # print(ctc_tgt_flat.device,ctc_lprobs.device)
        ctc_loss = F.ctc_loss(
            ctc_lprobs.transpose(0, 1),
            ctc_tgt_flat,
            ctc_lens,
            ctc_tgt_lens,
            reduction="sum",
            zero_infinity=True,
            blank=self.unit_vocab_size
        )
        ctc_loss /= ctc_tgt_lens.sum().item()
        return ctc_loss
        # except:
        # new_input_embeds, attention_mask, position_ids = self.upsample([x for x in text_embeddings] speech_tokens)
        # dummy_ctc_loss = self.llm_decoder(new_input_embeds).sum()*0.0
        # print(dummy_ctc_loss)
        # return dummy_ctc_loss
    
    def predict(self, tgt_text_reps, text):
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        tgt_text_reps = torch.concat([sos_eos_emb, tgt_text_reps, text_embedding, task_id_emb], dim=1)
        tgt_text_reps = self.input_proj(tgt_text_reps)
        tgt_text_reps = rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)

        text=text[:,:-1]
        text_len=text.size(1)      
        text_embedding = self.llm.model.model.embed_tokens(text)
        text_embedding = self.fusion(tgt_text_reps,text_embedding, None, None)
        
        new_input_embeds, attention_mask, position_ids = self.upsample(text_embedding)

        outputs=self.llm.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=new_input_embeds,
                labels=torch.full(new_input_embeds.shape[:2], IGNORE_INDEX, dtype=speech_tokens.dtype, device=speech_tokens.device),
                output_hidden_states=True,
                return_dict=True,
            )
        ctc_logits = self.llm_decoder(outputs.hidden_states[-1])
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_pred = ctc_lprobs.argmax(dim=-1).masked_fill_(~attention_mask.to(ctc_lprobs.device), self.unit_vocab_size)
        return ctc_pred