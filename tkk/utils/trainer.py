import collections
import json
import time
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterable
from typing import NamedTuple

import datasets
from datasets import load_metric
import numpy as np
import torch
import transformers.trainer_seq2seq
from torch.utils.data import Dataset
from packaging import version
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.file_utils import is_datasets_available, is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
)
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers.training_args import ParallelMode

from .training_arguments import WrappedSeq2SeqTrainingArguments

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]


def lmap(f: Callable, x: Iterable) -> List:
    return list(map(f, x))


def normalize(x: str) -> str:
    x = x.replace("[", " ")
    x = x.replace("]", " ")
    x = x.replace("group_by", "group by")
    x = x.replace("order_by", "order by")
    x = " ".join(x.strip().split())

    return x


def squad_metrics(p: EvalPrediction):
    metric = load_metric("./squad.py", cache_dir="./cache")
    return metric.compute(predictions=p.predictions, references=p.items)


class EvaluateFriendlySeq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
            self,
            evaluator,
            *args: WrappedSeq2SeqTrainingArguments,
            eval_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            wandb_run_dir: Optional[str] = None,
            task: Optional[str] = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.wandb_run_dir = wandb_run_dir
        self.task = task

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            eval_examples: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # print([eval_examples[idx]['arg_path'] for idx in range(len(eval_examples))])

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        # print("output in evaluate: ", output)

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                output.predictions,
                "eval_{}".format(self.state.epoch),
            )
            summary = self.compute_metrics(eval_preds, section="dev")
            output.metrics.update(summary)

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)
        
        with open(os.path.join(self.args.output_dir, "metrics.json"), "w") as f:
            json.dump(output.metrics, f, indent=4)

        for key in list(output.metrics.keys()):
            if "scores" in key:
                output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset],
            test_examples: Optional[Dataset],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                test_examples, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds, section="test"))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "no_repeat_ngram_size": 0,  # FIXME: hard coding the no_repeat_ngram_size
        }

        if "description_input_ids" in inputs:
            gen_kwargs["description_input_ids"] = inputs["description_input_ids"]
        if "description_attention_mask" in inputs:
            gen_kwargs["description_attention_mask"] = inputs["description_attention_mask"]
        if "knowledge_input_ids" in inputs:
            gen_kwargs["knowledge_input_ids"] = inputs["knowledge_input_ids"]
        if "knowledge_attention_mask" in inputs:
            gen_kwargs["knowledge_attention_mask"] = inputs["knowledge_attention_mask"]
        if "task_ids" in inputs:
            gen_kwargs["task_ids"] = inputs["task_ids"]

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _post_process_function(
            self, examples: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        assert isinstance(examples, Dataset)

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        predictions = lmap(normalize, predictions)

        # Save locally.
        if self.args.local_rank <= 0:
            with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(**{"prediction": predictions[idx]}, **examples[idx]) for idx in range(len(predictions))],
                    f,
                    indent=4,
                )

        if "main" in self.task:
            return EvalPrediction(predictions=predictions, items=[examples[idx] for idx in range(len(predictions))])
        else:
            predictions = [{"id": str(k), "prediction_text": v} for k, v in enumerate(predictions)]
            references = [{"id": str(k), "answers": {'answer_start': [0], 'text': [v]}} for k, v in
                                   enumerate([example["seq_out"] for example in examples])]
            return EvalPrediction(predictions=predictions, items=references)

    def _compute_metrics(self, eval_prediction: EvalPrediction, section) -> dict:
        if "main" in self.task:
            return self.evaluator.evaluate(eval_prediction.predictions, eval_prediction.items, section)
        else:
            return squad_metrics(eval_prediction)
