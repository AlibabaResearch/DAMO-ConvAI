from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DPOConfig:
    r"""
    Arguments pertaining to the PPO, DPO and KTO training.
    """

    beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    # pref_ftx: float = field(
    #     default=0.0,
    #     metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    # )
    pref_loss: Literal["sigmoid", "orpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )

    @property
    def use_ref_model(self):
        return self.pref_loss not in ["orpo"]
