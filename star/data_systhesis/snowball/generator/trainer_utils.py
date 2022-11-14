from typing import Dict, NamedTuple, Optional

import numpy as np

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
  original_logic: Optional[str]


class PredictionOutputWithSize(NamedTuple):
  predictions: np.ndarray
  predictions_size: np.ndarray
  label_ids: Optional[np.ndarray]
  label_size: Optional[np.ndarray]
  metrics: Optional[Dict[str, float]]