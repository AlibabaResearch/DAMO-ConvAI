import random, re, os
from functools import partial
from fastai.text.all import *
from hugdatafast.transform import CombineTransform
import json
import numpy as np

Max_len = 512
class MyConfig(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value

def adam_no_correction_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, **kwargs):
    p.data.addcdiv_(grad_avg, (sqr_avg).sqrt() + eps, value = -lr)
    return p

def Adam_no_bias_correction(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01, decouple_wd=True):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, adam_no_correction_step]
    return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)

def linear_warmup_and_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    if warmup_pct: warmup_steps = int(warmup_pct * total_steps)
    step_i = round(pct * total_steps)
    # According to the original source code, two schedules take effect at the same time, but decaying schedule will be neglible in the early time.
    decayed_lr = (lr_max-end_lr) * (1 - step_i/total_steps) ** decay_power + end_lr # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay
    warmed_lr = decayed_lr * min(1.0, step_i/warmup_steps) # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py#L44
    return warmed_lr

def linear_warmup_and_then_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    if warmup_pct: warmup_steps = int(warmup_pct * total_steps)
    step_i = round(pct * total_steps)
    if step_i <= warmup_steps: # warm up
        return lr_max * min(1.0, step_i/warmup_steps)
    else: # decay
        return (lr_max-end_lr) * (1 - (step_i-warmup_steps)/(total_steps-warmup_steps)) ** decay_power + end_lr

def load_part_model(file, model, prefix, device=None, strict=True):
    "assume `model` is part of (child attribute at any level) of model whose states save in `file`."
    distrib_barrier()
    if prefix[-1] != '.': prefix += '.'
    if isinstance(device, int): device = torch.device('cuda', device)
    elif device is None: device = 'cpu'
    state = torch.load(file, map_location=device)
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    # model_state = {k[len(prefix):] : v for k,v in model_state.items() if k.startswith(prefix) and '_sss' not in k}
    # for k, v in model_state.items():
    get_model(model).load_state_dict(model_state, strict=strict)

def load_model_(learn, files, device=None, **kwargs):
    "if multiple file passed, then load and create an ensemble. Load normally otherwise"
    merge_out_fc = kwargs.pop('merge_out_fc', None)
    if not isinstance(files, list): 
        learn.load(files, device=device, **kwargs)
        return
    if device is None: device = learn.dls.device
    model = learn.model.cpu()
    models = [model, *(deepcopy(model) for _ in range(len(files)-1)) ]
    for f,m in zip(files, models):
        file = join_path_file(f, learn.path/learn.model_dir, ext='.pth')
        load_model(file, m, learn.opt, device='cpu', **kwargs)
    learn.model = Ensemble(models, device, merge_out_fc)
    return learn

class ConcatTransform(CombineTransform):
    def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='text', book='multi'):
        super().__init__(hf_dset, in_cols=[text_col], out_cols=['input_ids', 'sentA_length'])
        self.max_length = max_length
        self.hf_tokenizer = hf_tokenizer
        self.book = book

    def reset_states(self):
        self.input_ids = [self.hf_tokenizer.cls_token_id]
        self.sent_lens = []

    def accumulate(self, sentence):
        if 'isbn' in sentence: return
        tokens = self.hf_tokenizer.convert_tokens_to_ids(self.hf_tokenizer.tokenize(sentence))
        tokens = tokens[:self.max_length-2] # trim sentence to max length if needed
        if self.book == 'single' or \
             (len(self.input_ids) + len(tokens) + 1 > self.max_length) or \
             (self.book == 'bi' and len(self.sent_lens)==2) :
            self.commit_example(self.create_example())
            self.reset_states()
        self.input_ids += [*tokens, self.hf_tokenizer.sep_token_id]
        self.sent_lens.append(len(tokens)+1)

    def create_example(self):
        if not self.sent_lens: return None
        self.sent_lens[0] += 1 # cls
        if self.book == 'multi':
            diff= 99999999
            for i in range(len(self.sent_lens)):
                current_diff = abs(sum(self.sent_lens[:i+1]) - sum(self.sent_lens[i+1:]))
                if current_diff > diff: break
                diff = current_diff
            return {'input_ids': self.input_ids, 'sentA_length': sum(self.sent_lens[:i])}
        else:
            return {'input_ids': self.input_ids, 'sentA_length': self.sent_lens[0]}

# Data Process

class ELECTRADataProcessor(object):
  """Given a stream of input text, creates pretraining examples."""

  def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='text', lines_delimiter='\n', minimize_data_size=True,
               apply_cleaning=True):
    self.hf_tokenizer = hf_tokenizer
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length

    self.hf_dset = hf_dset
    self.text_col = text_col
    self.lines_delimiter = lines_delimiter
    self.minimize_data_size = minimize_data_size
    self.apply_cleaning = apply_cleaning

  def map(self, **kwargs):
    "Some settings of datasets.Dataset.map for ELECTRA data processing"
    num_proc = kwargs.pop('num_proc', os.cpu_count())
    return self.hf_dset.my_map(
      function=self,
      batched=True,
      remove_columns=self.hf_dset.column_names,  # this is must b/c we will return different number of rows
      disable_nullable=True,
      input_columns=[self.text_col],
      writer_batch_size=10 ** 4,
      num_proc=num_proc,
      **kwargs
    )

  def __call__(self, texts):
    if self.minimize_data_size:
      new_example = {'input_ids': [], 'sim_label':[], 'ssf_label':[],'question_mask_plm':[],'column_mask_plm':[],'column_word_len':[],'position_ids':[],'sim_mask':[],'rtd_label': [],}
    for line in texts:  # for every doc
      
      example = self.add_line(line)
      if example:
        for k, v in example.items(): 
          if k=='input_ids' and len(v)==0:
            break
          new_example[k].append(v)
    return new_example

  def add_line(self, lines):
    """Adds a line of text to the current example being built."""
    # mlm
    line = json.loads(lines)

    input_ids = line['input_id']
    sim_label = line['sim_label']
    sim_mask = line['sim_mask']
    # for idx,item in enumerate(sim_mask):
    #     if item == True:
    #         sim_mask[idx] = False
    #         break
    ssf_label = line['ssf_label']
    question_mask_plm = line['question_mask_plm']
    rtd_label = line['rtd_label']
    column_mask_plm = line['column_mask_plm']
    column_word_len = line['column_word_len']
    position_ids = line['position_ids']
    if len(input_ids) < Max_len:
        input_ids = input_ids + [0]*(Max_len - len(input_ids))
        question_mask_plm = question_mask_plm + [1]*(Max_len - len(question_mask_plm))
        rtd_label = rtd_label + [0]*(Max_len - len(rtd_label))
        column_mask_plm = column_mask_plm + [0]*(Max_len - len(column_mask_plm))
        position_ids = position_ids + list(range(len(position_ids),Max_len))

    if self.minimize_data_size:
      return {
        'input_ids': input_ids,
        'sim_label': sim_label,
        'ssf_label': ssf_label,
        'question_mask_plm': question_mask_plm,
        'column_mask_plm': column_mask_plm,
        'column_word_len': column_word_len,
        'position_ids': position_ids,
        'sim_mask': sim_mask,
        'rtd_label': rtd_label,
      }
