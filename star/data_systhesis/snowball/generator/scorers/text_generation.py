import json
import torch

from bleu import list_bleu
import os

def is_rank_0():
  if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
      return True
  else:
    return True
  return False

class TextGenerationScorer:
  def __init__(self, tokenizer, bos_id, eos_id, output_path):
    self.bos_id = bos_id
    self.eos_id = eos_id
    self.output_path = output_path
    self.tokenizer = tokenizer

  def __call__(self, prediction, epoch = 0, snow_ball = False, mode_name='eval'):
    
    epoch = 0 if not epoch else epoch
    
    if snow_ball:
        output_path = self.output_path + 'augmentation.json'
    else:
        output_dir = os.path.join(self.output_path, mode_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = output_dir + os.sep + 'epoch_{}'.format(int(epoch)) + '.json'
    
    preds = prediction.predictions
    preds_size = prediction.predictions_size
    label_ids = prediction.label_ids
    label_size = prediction.label_size
    logics = prediction.logics
    original_logic = prediction.original_logic
    
    p_start, l_start = 0, 0
    correct, total = 0, 0
    ref = []
    hyp = []
    if is_rank_0():
      fout = open(output_path, "w")
    for idx, (p_size, l_size) in enumerate(zip(preds_size, label_size)):
      p_end = p_start + p_size
      l_end = l_start + l_size
      pred = self.get_sequence(preds[p_start: p_end])
      label = self.get_sequence(label_ids[l_start: l_end])
      p_start = p_end
      l_start = l_end
      if pred == label:
        correct += 1
      total += 1
      if is_rank_0():
        pred_text = self.tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        label_text = self.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        ref.append(label_text)
        hyp.append(pred_text)
        if snow_ball:
            fout.write(
              json.dumps({
                "idx": idx,
                "mutated_logic": logics[idx],
                "mutated_text": pred_text,
                "original_logic":original_logic[idx],
                "original_text": label_text}) + "\n")
            
        else:
            fout.write(
              json.dumps({
                "idx": idx,
                "logic": logics[idx],
                "pred": pred_text,
                "label": label_text}) + "\n")
    # score = list_bleu([ref], hyp, tmp_dir='tmp/tmp_bleu')
    score = list_bleu([ref], hyp)
    return {
      "bleu": score,
      "accuracy": correct / total,
      "correct": correct,
      "total": total
    }


  def get_sequence(self, seq):
    processed_seq = []
    for idx in seq:
      if idx == self.bos_id:
        continue
      if idx == self.eos_id:
        break
      processed_seq.append(int(idx))
    return processed_seq




