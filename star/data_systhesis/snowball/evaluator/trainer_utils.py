import os
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch

class EvalPredictionWithSize(NamedTuple):
  """
  Evaluation output (always contains labels), to be used
  to compute metrics.
  """

  predictions: np.ndarray
  predictions_size: np.ndarray
  label_ids: np.ndarray
  label_size: np.ndarray
  logics: Optional[str]


class PredictionOutputWithSize(NamedTuple):
  predictions: np.ndarray
  predictions_size: np.ndarray
  label_ids: Optional[np.ndarray]
  label_size: Optional[np.ndarray]
  metrics: Optional[Dict[str, float]]
    
    


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, output_dir='.'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.1
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.auc_score_max = 0
        self.delta = delta
        self.output_dir = output_dir

    def __call__(self, auc_score, model):

        score = auc_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(auc_score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}. The best AUC score was {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint( auc_score, model)
            self.counter = 0

    def save_checkpoint(self, auc_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation AUC score ({self.auc_score_max:.6f} --> {auc_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.output_dir,'checkpoint.pt'))	# 这里会存储迄今最优模型的参数
        self.auc_score_max = auc_score
