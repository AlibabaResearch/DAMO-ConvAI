from dataclasses import dataclass

@dataclass
class ResamplingMethod:
    n_resamples: int
    batch: int | None
