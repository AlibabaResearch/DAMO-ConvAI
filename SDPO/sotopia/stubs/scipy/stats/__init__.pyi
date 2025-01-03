from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike
from scipy.stats._resampling import ResamplingMethod
from scipy.stats._result_classes import PearsonRResult

@dataclass
class SignificanceResult:
    statistic: float
    pvalue: float
    correlation: float

def pearsonr(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = ...,
    method: ResamplingMethod | None = ...,
) -> PearsonRResult: ...
def spearmanr(
    a: ArrayLike,
    b: ArrayLike | None = ...,
    axis: int | None = ...,
    nan_policy: Literal["propagate", "raise", "omit"] | None = ...,
    alternative: Literal["two-sided", "less", "greater"] | None = ...,
) -> SignificanceResult: ...
