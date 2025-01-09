import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen2ForCausalLM
from openomni.constants import IGNORE_INDEX


# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Label smoothing module."""

import torch
from torch import nn

import torch
from einops import rearrange, repeat
from torch import einsum, nn
import torch.nn.functional as F

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

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        # print(x.shape,target.shape)
        x = x.reshape(-1, self.size)
        target = target.reshape(-1)
        # print(x.shape,target.shape)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

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

def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()

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

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

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
        self.upsample_factor = config.ctc_upsample_factor
        self.unit_vocab_size = config.unit_vocab_size
        qwen_tiny_path="/mnt/workspace/lr/datasets/checkpoints/Qwen/Qwen2.5-0.5B-Instruct"
        self.llm = Qwen2Encoder(qwen_tiny_path)
        
        self.n_dims=896 # fix

        self.llm_input_size = self.n_dims
        self.llm_output_size = self.n_dims

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, self.llm_input_size)
        self.llm_decoder = nn.Linear(self.llm_output_size, self.unit_vocab_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.unit_vocab_size + 3,
            padding_idx=IGNORE_INDEX,
            smoothing=0,
            normalize_length=True,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(self.unit_vocab_size + 3, self.llm_input_size)

        # 4. sampling method
        self.sampling = ras_sampling

        modules = [nn.Linear(config.hidden_size, self.n_dims*4)]
        modules.append(nn.GELU())
        modules.append(nn.Linear(self.n_dims*4, self.n_dims*4))

        self.input_proj = nn.Sequential(*modules)

        self.fusion=RLSIAT(self.n_dims)


    def sampling_ids(
            self,
            weighted_scores,
            decoded_tokens,
            sampling,
            ignore_eos
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.unit_vocab_size not in top_ids):
                break
        return top_ids

    def forward(self, text_reps, text_labels, speech_tokens, text_tokens=None, embedding=None):

        # if text_tokens is None:
        tgt_text_reps = []
        for text_rep, text_label in zip(text_reps, text_labels):  
            tgt_text_reps.append(text_rep[text_label.ne(IGNORE_INDEX)])
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
            
        speech_token_lens = speech_tokens.ne(IGNORE_INDEX).long().sum(dim=-1)

        speech_embeddings = self.speech_embedding(torch.nn.utils.rnn.pad_sequence([x[x.ne(IGNORE_INDEX)] for x in speech_tokens], batch_first=True, padding_value=0))

        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        lm_target = [torch.tensor([IGNORE_INDEX] * (1+ text_token_lens[i]+1) + speech_tokens[i, :speech_token_lens[i]].tolist() +
                                  [self.unit_vocab_size],dtype=torch.long).to(speech_tokens.device) for i in range(speech_tokens.size(0))]

        lm_input=[]

        for batch_id,(text_token_len, speech_token_len) in enumerate(zip(text_token_lens,speech_token_lens)):
            lm_input.append(torch.cat([sos_eos_emb.squeeze(dim=0),
            text_embeddings[batch_id][:text_token_len],
            task_id_emb.squeeze(dim=0),
            speech_embeddings[batch_id][:speech_token_len],
            sos_eos_emb.squeeze(dim=0)]))
        
        new_input_embeds=lm_input
        new_labels=lm_target

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []

        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels_padded.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=new_labels_padded.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_labels = new_labels_padded

        outputs=self.llm.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=new_input_embeds,
                labels=new_labels,
                output_hidden_states=True,
                return_dict=True,
        )

        logits = self.llm_decoder(outputs.hidden_states[-1])
        loss = self.criterion_ce(logits[:,:-1], new_labels[:,1:])
        return loss
    
    def predict(self, 
            tgt_text_reps, 
            text,
            sampling= 25,
            max_token_text_ratio = 20,
            min_token_text_ratio = 2,
            **kwargs):
 
        tgt_text_reps =  self.input_proj(tgt_text_reps)
        tgt_text_reps=rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
        # text=torch.tensor([text], dtype=torch.long).to(tgt_reps[0].device)
        # drop the eos token
        text=text[:,:-1]
        text_len=text.size(1)      
        text_embedding = self.llm.model.model.embed_tokens(text)
        text_embedding = self.fusion(tgt_text_reps,text_embedding, None,None)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(text.device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        lm_input = torch.concat([sos_eos_emb, embedding, text_embedding, task_id_emb], dim=1)

        # 4. cal min/max_length
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)

        # print(min_len,max_len)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                    masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                    cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.unit_vocab_size:
                break
            if top_ids > self.unit_vocab_size:
                continue
            # in stream mode, yield token one by one
            # yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            
        return ' '.join([str(x) for x in out_tokens])

    # @torch.inference_mode()
    # def inference(
    #         self,
    #         text=None,
    #         sampling= 25,
    #         max_token_text_ratio = 20,
    #         min_token_text_ratio = 2,
    # ):
    #     texts='''四是四，十是十，十四是十四，四十是四十。
    #         黑化肥发灰，灰化肥发黑，黑化肥发灰会挥发，灰化肥挥发会发黑。
    #         吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。
    #         八百标兵奔北坡，炮兵并排北边跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。
    #         红凤凰，黄凤凰，粉红凤凰，花凤凰。
    #         牛郎年年恋刘娘，刘娘念念恋牛郎。
    #         高高山上一棵葡萄树，树上有条大蟒蛇，蛇要咬葡萄，葡萄要砸蛇。
    #         松树松，柏树柏，松树柏树都是树。
    #         妈妈骑马，马慢，妈妈骂马。
    #         六十六岁的陆六，摔了六十六个溜溜球。
    #         河里有只白毛鹅，白毛鹅在划水，水面水花四面散，水花洒在白毛鹅身上。
    #         公公背着公公公，婆婆抱着婆婆婆，公公婆婆忙着抱背公公婆婆。
    #         床前明月光，月光照床前，床前有只羊，羊在床前望。
    #         绿领巾，红领巾，绿领巾围脖上，红领巾脖上绕。
    #         大花碗，装大碗饭，饭装满了大花碗，碗中饭香满院。
    #         白猫黑猫，一只大一只小，小猫说大猫大，大猫说小猫小。
    #         酸枣树上酸枣多，酸枣树下小孙坐，小孙笑说酸枣酸，酸枣甜时我来坐。
    #         东边一条河，西边一座山，河上有桥山连山，河里鱼儿一串串。
    #         风吹藤，藤挂灯，灯照藤，藤影晃灯影动。
    #         树上有只小黄鹂，树下坐着小黄梨，小黄鹂叽叽叫，小黄梨低头笑。'''

    #     texts='''She sells seashells by the seashore.
    #         Peter Piper picked a peck of pickled peppers.
    #         How much wood would a woodchuck chuck if a woodchuck could chuck wood?
    #         Betty Botter bought some butter, but she said the butter's bitter.
    #         A big black bear sat on a big black rug.
    #         Red lorry, yellow lorry, red lorry, yellow lorry.
    #         Six slippery snails slid slowly seaward.
    #         Fuzzy Wuzzy was a bear, Fuzzy Wuzzy had no hair, Fuzzy Wuzzy wasn’t very fuzzy, was he?
    #         I saw Susie sitting in a shoeshine shop.
    #         Which witch is which?
    #         Can you can a can as a canner can can a can?
    #         I have got a date at a quarter to eight; I'll see you at the gate, so don't be late.
    #         If two witches were watching two watches, which witch would watch which watch?
    #         Green glass globes glow greenly.
    #         Nine nimble noblemen nimbly nibbling nuts.
    #         The thirty-three thieves thought that they thrilled the throne throughout Thursday.
    #         Six sleek swans swam swiftly southwards.
    #         Lesser leather never weathered wetter weather better.
    #         If a dog chews shoes, whose shoes does he choose?
    #         How can a clam cram in a clean cream can?'''
    #     texts=texts.split('\n')
    #     xx=[]
    #     for text in texts:
    #         # text="四是四，十是十，十四是十四，四十是四十"
    #         text=text.strip()
    #         print(text)
    #         text= self.small_tokenizer.encode(text)
    #         text_len = len(text)
    #         text=torch.tensor([text], dtype=torch.long).to('cuda')
    #         device = text.device
            
    #         text = self.llm.model.model.embed_tokens(text)

    #         # 2. encode embedding
    #         embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

    #         # 3. concat llm_input
    #         sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
    #         task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

    #         lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb], dim=1)

    #         # 4. cal min/max_length
    #         min_len = int(text_len * min_token_text_ratio)
    #         max_len = int(text_len * max_token_text_ratio)

    #         # print(min_len,max_len)

    #         # 5. step by step decode
    #         out_tokens = []
    #         cache = None
    #         for i in range(max_len):
    #             y_pred, cache = self.llm.forward_one_step(lm_input,
    #                                                     masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
    #                                                     cache=cache)
    #             logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
    #             top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
    #             if top_ids == self.unit_vocab_size:
    #                 break
    #             if top_ids > self.unit_vocab_size:
    #                 continue
    #             # in stream mode, yield token one by one
    #             # yield top_ids
    #             out_tokens.append(top_ids)
    #             lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
    #         xx.append(out_tokens)
    #     print(xx)
    #     return out_tokens