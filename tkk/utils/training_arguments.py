from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments


@dataclass
class WrappedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use a `sortish sampler` or not. Only possible if the underlying datasets are `Seq2SeqDataset` for
        now but will become generally available in the near future.

        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    cfg: str = None
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    input_max_length: int = field(
        default=512, metadata={
            "help": "The sequence max_length we feed into the model, the rest part will be truncated and dropped."}
    )
    generation_max_length: int = field(
        default=512, metadata={
            "help": "The max_length to use on each evaluation loop when predict_with_generate=True."
                    " Will default to the max_length value of the model configuration."}
    )
    generation_num_beams: int = field(
        default=2, metadata={
            "help": "The num_beams to use on each evaluation loop when predict_with_generate=True."
                    " Will default to the num_beams value of the model configuration."}
    )
    load_weights_from: Optional[str] = field(
        default=None, metadata={
            "help": "The directory to load the model weights from."}
    )
    prompt_initialization: Optional[bool] = field(
        default=False, metadata={
            "help": "Wether to initialize the prompts."} 
    )
