from dataclasses import dataclass,field
from torch.utils import data
from transformers import (

    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForPreTraining,
)

from . import register_argumentclass

from transformers.models.auto import configuration_auto
from transformers.models.auto import tokenization_auto
from transformers.models.bert.tokenization_bert import BertTokenizer


class BaseArguments:
    def process(self):
        #return {"smile":"^_^"}
        return {}

@register_argumentclass("model_seq2seq")
@dataclass
class Seq2SeqArguments(BaseArguments):
#Generation
    auto_model = AutoModelForSeq2SeqLM

    mask_input: bool = field(default=True)
    mask_rate: float = field(default=0)

    neftune_alpha: float=field(default=0)
    token_dropout: float=field(default=0)
    only_drop_target: bool=field(default=False)
    neft_all_token: bool=field(default=False)

    drop_attention_mask: bool = field(default=False)

    token_noise: bool=field(default=False)
    token_noise_hard: bool=field(default=False)
    token_noise_sample: bool=field(default=False)
    token_noise_random: bool=field(default=False)
    mixup: bool=field(default=False)

    replace: bool=field(default=False)
    replace_rate: float=field(default=0)

    temperature: float = field(default=1)

    not_mask_source: bool=field(default=True)
    not_mask_tgt: bool=field(default=False)






