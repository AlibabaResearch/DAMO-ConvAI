from multiprocessing import context
import os, sys, random
from pathlib import Path
from functools import partial
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import datasets
from fastai.text.all import *
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM #, ElectraForPreTraining
from electra_for_inbatch import ElectraForPreTraining
from _utils.hugdatafast import *
from _utils.utils_inbatch import *
from _utils.would_like_to_pr import *

from torch.nn.functional import gelu
from transformers.activations import get_activation
from sklearn.metrics import accuracy_score
from _utils.model_utils import PoolingFunction,lens2mask,lens2mask2
from torch.nn import CrossEntropyLoss
BertLayerNorm = torch.nn.LayerNorm
COLUMN_SQL_LABEL_COUNT = 383

## 1. Configuraton
c = MyConfig({
    'device': 'cuda:0',
    # 'device': 'cpu',
    
    'base_run_name': 'vanilla',
    'seed': 11081,

    'adam_bias_correction': False,
    'schedule': 'original_linear',
    'sampling': 'fp32_gumbel',
    'electra_mask_style': True,
    'gen_smooth_label': False,
    'disc_smooth_label': False,

    'size': 'large',
    'datas': ['score'],
    
    # 'logger': 'wandb',
    'logger': None,
    'num_workers': 3,
})

## Check and Default
assert c.sampling in ['fp32_gumbel', 'fp16_gumbel', 'multinomial']
assert c.schedule in ['original_linear', 'separate_linear', 'one_cycle', 'adjusted_one_cycle']
for data in c.datas: assert data in ['wikipedia', 'bookcorpus', 'openwebtext', 'score']
assert c.logger in ['wandb', 'neptune', None, False]
if not c.base_run_name: c.base_run_name = str(datetime.now(timezone(timedelta(hours=+8))))[6:-13].replace(' ','').replace(':','').replace('-','')
if not c.seed: c.seed = random.randint(0, 999999)
c.run_name = f'{c.base_run_name}_{c.seed}'
if c.gen_smooth_label is True: c.gen_smooth_label = 0.1
if c.disc_smooth_label is True: c.disc_smooth_label = 0.1

## electra sizes setting
## Setting of different sizes
c.mask_prob = 0.15
c.lr = 1e-6
c.bs = 80
c.steps = 100*1000 
c.max_length = 512 
disc_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-discriminator')
gen_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-generator')

## note that public electra-small model is actually small++ and don't scale down generator size 
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{c.size}-generator")

## logger
if c.logger == 'wandb':
  import wandb
  from fastai.callback.wandb import  WandbCallback

## Path to data
Path('./datasets', exist_ok=True)
Path('./checkpoints/pretrain').mkdir(exist_ok=True, parents=True)

## 1. Load Data
dsets = []
ELECTRAProcessor = partial(ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=c.max_length)

## Wikipedia
if 'score' in c.datas:
  print('load/download SCoRE Corpus')
  sc = datasets.load_dataset('text',data_files='datasets/alltask_final.txt')['train']
  print('load/create data from SCoRE Corpus for ELECTRA')
  e_sc = ELECTRAProcessor(sc, apply_cleaning=False).map(cache_file_name=f"electra_sc_{c.max_length}.arrow", num_proc=1)  # cache
  # e_sc = ELECTRAProcessor(sc, apply_cleaning=False).map(num_proc=1)  # no cache
  dsets.append(e_sc)

assert len(dsets) == len(c.datas)

merged_dsets = {'train': datasets.concatenate_datasets(dsets)}
## HF_Datasets: a dataset class in fastai
hf_dsets = HF_Datasets(merged_dsets, cols={'input_ids':noop,'sim_label':noop,'ssf_label': noop, 'question_mask_plm':noop, 'column_mask_plm':noop, 'column_word_len':noop, 'position_ids':noop, 'sim_mask':noop, 'rtd_label':noop},
                       hf_toker=hf_tokenizer, n_inp=11)
          
dls = hf_dsets.dataloaders(bs=c.bs, num_workers=c.num_workers, pin_memory=False,
                           shuffle_train=False,
                           srtkey_fc=False, 
                           pad_idx = False,
                           cache_dir='./datasets/electra_dataloader', cache_name='dl_{split}.json')


## 2. Training Objectives
## 2. Masked language model objective
"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). 
It is a little bit faster cuz 
- intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- don't iterate the tensor when create special tokens mask
And
- doesn't require huggingface tokenizer
- cost you only 550 µs for a (128,128) tensor on gpu, so dynamic masking is cheap   
"""
## 2.1 MLM objective callback

def mask_tokens(inputs, question_mask_plm, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
  ## inputs: input_ids
  """ 
  Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence. 
  * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
  """

  device = inputs.device
  labels = inputs.clone()
  
  cls_id = hf_tokenizer.cls_token_id

  ## Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
  ## mlm probability matrix
  probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
  ## special tokens
  special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
  for sp_id in special_token_indices:
    special_tokens_mask = special_tokens_mask | (inputs==sp_id)
  special_tokens_mask = special_tokens_mask | question_mask_plm
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  ## mlm probability matrix (0 or 1)
  mlm_mask = torch.bernoulli(probability_matrix).bool()
  labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens (not special tokens)

  ## mask operation
  ## mask  (mlm_probability * (1-replace_prob-orginal_prob))
  mask_prob = 1 - replace_prob - orginal_prob
  mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
  inputs[mask_token_mask] = mask_token_index

  ## replace operation
  ## replace with a random token (mlm_probability * replace_prob)
  if int(replace_prob)!=0:
    rep_prob = replace_prob/(replace_prob + orginal_prob)
    replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[replace_token_mask] = random_words[replace_token_mask]


  ## do nothing (mlm_probability * orginal_prob)
  pass

  return inputs, labels, mlm_mask

class MaskedLMCallback(Callback):
  @delegates(mask_tokens)
  def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, for_electra=True, **kwargs):
    self.ignore_index = ignore_index
    self.for_electra = for_electra
    self.mask_tokens = partial(mask_tokens,
                               mask_token_index=mask_tok_id,
                               special_token_indices=special_tok_ids,
                               vocab_size=vocab_size,
                               ignore_index=-100,
                               **kwargs)
  
  ## called at the beginning of each batch, just after drawing said batch.
  def before_batch(self):
    input_ids, sim_label, ssf_label, question_mask_plm, column_mask_plm, column_word_len, position_ids, column_word_num, sim_mask_p, sim_mask_n, rtd_label  = self.xb
    
    input_ids_drop = input_ids.clone()
    masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids, question_mask_plm)
    
    if self.for_electra:
      self.learn.xb = (masked_inputs, input_ids_drop, is_mlm_applied, labels, question_mask_plm, column_mask_plm, column_word_len, position_ids, column_word_num)
      self.learn.yb = (labels, sim_label, sim_mask_p, sim_mask_n, ssf_label, rtd_label)
    else:
      self.learn.xb, self.learn.yb = (masked_inputs), (labels,)

mlm_cb = MaskedLMCallback(mask_tok_id=hf_tokenizer.mask_token_id, 
                          special_tok_ids=hf_tokenizer.all_special_ids, 
                          vocab_size=hf_tokenizer.vocab_size,
                          mlm_probability=c.mask_prob,
                          replace_prob=0.0 if c.electra_mask_style else 0.1, 
                          orginal_prob=0.15 if c.electra_mask_style else 0.1,
                          for_electra=True)

class ELECTRAModel(nn.Module):

  def __init__(self, generator, discriminator, hf_tokenizer):
    super().__init__()
    self.generator, self.discriminator = generator,discriminator
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
    self.hf_tokenizer = hf_tokenizer


  def to(self, *args, **kwargs):
    "Also set dtype and device of contained gumbel distribution if needed"
    super().to(*args, **kwargs)
    a_tensor = next(self.parameters())
    device, dtype = a_tensor.device, a_tensor.dtype
    if c.sampling=='fp32_gumbel': dtype = torch.float32
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

  def forward(self, masked_inputs, input_ids_drop, is_mlm_applied, labels, question_mask_plm, column_mask_plm, column_word_len, position_ids, column_word_num):
    """
    masked_inputs (Tensor[int]): (B, L)
    sentA_lenths (Tensor[int]): (B, L)
    is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
    labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
    """
    attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs)
    gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids, position_ids=position_ids)[0] # (B, L, vocab size)
    ## reduce size to save space and speed
    mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)
    
    with torch.no_grad():
      ## sampling
      pred_toks = self.sample(mlm_gen_logits) # ( #mlm_positions, )
      ## produce inputs for discriminator
      generated = masked_inputs.clone() # (B,L)
      generated[is_mlm_applied] = pred_toks # (B,L)
      ## produce labels for discriminator
      is_replaced = is_mlm_applied.clone() # (B,L)
      is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)
      is_replaced = is_replaced
    
    _, col_logits_drop, context_logits_drop = self.discriminator(input_ids_drop, attention_mask, token_type_ids, position_ids=position_ids,column_mask_plm=column_mask_plm,column_word_len=column_word_len,column_word_num=column_word_num)
    
    disc_logits, col_logits, context_logits = self.discriminator(generated, attention_mask, token_type_ids, position_ids=position_ids,column_mask_plm=column_mask_plm,column_word_len=column_word_len,column_word_num=column_word_num)
    
    return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied, col_logits, context_logits, col_logits_drop, context_logits_drop

  def _get_pad_mask_and_token_type(self, input_ids):
    """
    Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
    """
    attention_mask = input_ids != self.hf_tokenizer.pad_token_id
    seq_len = input_ids.shape[1]
    token_type_ids = torch.tensor([ ([0]*seq_len) for _ in range(input_ids.shape[0])],  
                                  device=input_ids.device)
    return attention_mask, token_type_ids

  def sample(self, logits):
    "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
    if c.sampling == 'fp32_gumbel':
      gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
      return (logits.float() + gumbel).argmax(dim=-1)
    elif c.sampling == 'fp16_gumbel':  # 5.06 ms
      gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
      return (logits + gumbel).argmax(dim=-1)
    elif c.sampling == 'multinomial':  # 2.X ms
      return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()

class AutomaticWeightedLoss(nn.Module):
  """automatically weighted multi-task loss
  Params:
      num: int,the number of loss
      x: multi-task loss
  Examples:
      loss1=1
      loss2=2
      awl = AutomaticWeightedLoss(2)
      loss_sum = awl(loss1, loss2)
  """

  def __init__(self, num=2):
    super(AutomaticWeightedLoss, self).__init__()
    params = torch.ones(num, requires_grad=True)
    self.params = torch.nn.Parameter(params)

  def forward(self, *x):
    loss_sum = 0
    for i, loss in enumerate(x):
      loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
    return loss_sum

class Similarity(nn.Module):
  def __init__(self):
    super().__init__()
    self.cos = nn.CosineSimilarity(dim=-1)

  def forward(self, x, y):
    return self.cos(x,y)

class ELECTRALoss():
  def __init__(self, loss_weights=(1.0, 5.0), gen_label_smooth=False, disc_label_smooth=False, weight = ([0.025] + [1] * 382)):
    self.loss_weights = loss_weights
    self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLossFlat()
    self.disc_loss_fc = nn.BCEWithLogitsLoss()
    self.context_loss_fct = nn.BCEWithLogitsLoss()
    self.awl = AutomaticWeightedLoss(3)
    self.weights = torch.tensor(weight).float().cuda()
    self.weighted_loss_fct = CrossEntropyLoss(weight=self.weights)
    self.disc_label_smooth = disc_label_smooth
    self.sim = Similarity()
    self.softmax = nn.Softmax(dim = 1)
    self.alpha = torch.tensor([3]).float().cuda()
    self.count = 0

  def __call__(self, pred, targ_ids, sim_label, sim_mask_p, sim_mask_n, ssf_label, rtd_label):
    mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied, col_logits, context_logits, col_logits_drop, context_logits_drop = pred
    gen_loss = self.gen_loss_fc(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
    device = generated.device
    
    ## schema prediction
    is_replaced = is_replaced | rtd_label  ## 
    
    disc_logits = disc_logits.masked_select(non_pad) # -> 1d tensor
    is_replaced = is_replaced.masked_select(non_pad) # -> 1d tensor
    if self.disc_label_smooth:
      is_replaced = is_replaced.float().masked_fill(~is_replaced, self.disc_label_smooth)
    disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())

    ## sst loss
    col_loss = self.weighted_loss_fct(col_logits.view(-1, COLUMN_SQL_LABEL_COUNT), ssf_label.view(-1))
    col_loss_drop = self.weighted_loss_fct(col_logits_drop.view(-1, COLUMN_SQL_LABEL_COUNT), ssf_label.view(-1))
    col_loss = (col_loss + col_loss_drop)/2
    
    ## udt loss
    sim_logits = self.sim(context_logits_drop.unsqueeze(1), context_logits.unsqueeze(0))
    sim_logits_witht = sim_logits / 0.05
    exp_logits = torch.exp(sim_logits_witht) * sim_mask_n
    log_prob = sim_logits_witht - torch.log(exp_logits.sum(1, keepdim=True))
    an = sim_label * self.alpha + (1 - sim_mask_p) * (-1e20)
    step = self.softmax(an)
    mean_log_prob_pos = - (step * log_prob).sum(1) 
    context_loss = mean_log_prob_pos.mean()

    ## mlm loss
    electra_loss = gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]

    ## total loss
    total_loss =  self.awl(electra_loss, col_loss, context_loss)

    ## evaluate train acc

    if c.logger=='wandb':
      wandb.log({
          "total_loss": total_loss,
          "electra_loss": electra_loss,
          "col_loss": col_loss,
          "context_loss": context_loss,
      })
    self.count += 1
    return total_loss

## 5. Train
## Seed & PyTorch benchmark
torch.backends.cudnn.benchmark = True
dls[0].rng = random.Random(c.seed) # for fastai dataloader
random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)

## Generator and Discriminator
generator = ElectraForMaskedLM.from_pretrained(f'google/electra-{c.size}-generator')
discriminator = ElectraForPreTraining.from_pretrained(f'google/electra-{c.size}-discriminator', output_hidden_states=True)
discriminator.electra.embeddings = generator.electra.embeddings
generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

## ELECTRA training loop
electra_model = ELECTRAModel(generator, discriminator, hf_tokenizer)
electra_loss_func = ELECTRALoss(gen_label_smooth=c.gen_smooth_label, disc_label_smooth=c.disc_smooth_label)

## Optimizer
if c.adam_bias_correction: opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
else: opt_func = partial(Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)

## Learning rate shedule
if c.schedule.endswith('linear'):
  lr_shed_func = linear_warmup_and_then_decay if c.schedule=='separate_linear' else linear_warmup_and_decay
  lr_shedule = ParamScheduler({'lr': partial(lr_shed_func,
                                             lr_max=c.lr,
                                             warmup_steps=10000,  # 10k steps
                                             total_steps=c.steps,)})

## Learner
dls.to(torch.device(c.device))
## Learner: a class in fastai.text
learn = Learner(dls, electra_model,
                loss_func=electra_loss_func,
                opt_func=opt_func ,
                path='./checkpoints',
                model_dir='pretrain',
                ## cbs: one or a list of Callbacks to pass to the Learner
                cbs=[mlm_cb,
                    RunSteps(c.steps, [0.4], c.run_name+"_{percent}"),  # 100k steps
                     ],
                )

## multi-gpu training
learn.model = torch.nn.DataParallel(learn.model, device_ids=[0,1,2,3])

## logging
if c.logger == 'wandb':
  wandb.init(name=c.run_name, project='new_data', config={**c})
  learn.add_cb(WandbCallback(log_preds=False, log_model=False))

## Mixed precison and Gradient clip
# learn.to_native_fp16(init_scale=2.**11)
learn.to_fp16(init_scale=2.**11)
learn.add_cb(GradientClipping(1.))

## Print time and run name
print(f"{c.run_name} , starts at {datetime.now()}")

## Run
# learn.fit_one_cycle(n_epoch, lr_max)
if c.schedule == 'one_cycle': learn.fit_one_cycle(9999, lr_max=c.lr)
elif c.schedule == 'adjusted_one_cycle': learn.fit_one_cycle(9999, lr_max=c.lr, div=1e5, pct_start=10000/c.steps)
else: learn.fit(9999, cbs=[lr_shedule])
