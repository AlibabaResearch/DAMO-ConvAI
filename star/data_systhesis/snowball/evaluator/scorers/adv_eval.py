import json
import torch
import os

from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc,recall_score,precision_score

def is_rank_0():
  if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
      return True
  else:
    return True
  return False

class EvalScorer:
  def __init__(self, tokenizer, bos_id, eos_id, output_path):
    self.output_path = output_path
 
  #compute the Precision, Recall, F1 and AUC
  def compute_score(self, scores, labels,threshold=0.5):
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    accuracy, precision, recall, f1, auc_score = 0.0, 0.0, 0.0, 0.0, 0.0
    scores = scores.cpu()
    labels = labels.cpu()
    predictions = scores > threshold
    predictions = predictions.long()

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    return accuracy, precision, recall, f1, auc_score
    
  def __call__(self, prediction, epoch = 0, dump_output=True, mode_name='eval'):
    output_dir = os.path.join(self.output_path, mode_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = output_dir + os.sep + 'epoch_{}'.format(int(epoch)) + '.json'
    
    acc, p, r, f1, auc = self.compute_score(prediction["pred_scores"], prediction["pred_labels"])
    if dump_output:
        if is_rank_0():
          fout = open(output_path, "w")
        for idx, (logic, text, remark, label, score) in enumerate(zip(prediction["logics"], prediction["texts"], prediction["remarks"], prediction["pred_labels"], prediction["pred_scores"])):
          if is_rank_0():
            fout.write(
              json.dumps({
                    "logic":logic,
                    "text":text,
                    "remark":remark,
                    "label":str(label.item()),
                    "score":str(score.item())
              }) + '\n'
            )
        
    return {
            "eval_accuracy":acc,
            "eval_precision":p,
            "eval_recall" :r,
            "eval_F1" : f1,
            "eval_AUC":auc
    }

   





