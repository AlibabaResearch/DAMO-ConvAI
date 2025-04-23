from dataclasses import dataclass
from typing import Any, NamedTuple

@dataclass
class PearsonRResult:
    statistic: float
    pvalue: float
    correlation: float

    def confidence_interval(
        self, confidence_level: float = ..., method: Any = ...
    ) -> NamedTuple: ...
