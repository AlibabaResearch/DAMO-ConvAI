#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Dict, Tuple, Any, Optional
from torch.cuda.amp import autocast

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AdamW,
    TrainerCallback,
)

from transformers.trainer import *

from uie.seq2seq.constraint_decoder import get_constraint_decoder

import learn2learn as l2l

import random
import pdb

@dataclass
class ConstraintSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    save_better_checkpoint: bool = field(default=False,
                                         metadata={"help": "Whether to save better metric checkpoint"})
    start_eval_step: int = field(default=0, metadata={"help": "Start Evaluation after Eval Step"})
    trainer_type: str = field(default="meta_pretrain", metadata={"help": "Trainer for training model, containing meta_pretrain, meta_finetune, origin"})

class OriginalConstraintSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, decoding_type_schema=None, task='event', decoding_format='tree', source_prefix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema

        # Label smoothing by sum token loss, different from different Label smootheing
        if self.args.label_smoothing_factor != 0:
            print('Using %s' % self.label_smoother)
        else:
            self.label_smoother = None

        if self.args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             decoding_schema=self.decoding_format,
                                                             source_prefix=source_prefix,
                                                             task_name=task)
        else:
            self.constraint_decoder = None

        self.oom_batch = 0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        oom = False
        oom_message = ""
        try:
            loss = super().training_step(model, inputs)
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
                oom_message = str(e)
                logger.warning(f'ran out of memory {self.oom_batch} on {self.args.local_rank}')
                for k, v in inputs.items():
                    print(k, v.size())
            else:
                raise e

        if oom:
            self.oom_batch += 1
            raise RuntimeError(oom_message)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            **kwargs
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        if self.args.start_eval_step > 0 and self.state.global_step < self.args.start_eval_step:
            return

        previous_best_metric = self.state.best_metric
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        # Only save the checkpoint better than previous_best_metric
        if self.args.save_better_checkpoint and self.args.metric_for_best_model is not None:
            if metrics is not None and previous_best_metric is not None:
                if metrics[self.args.metric_for_best_model] <= previous_best_metric:
                    return

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

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

        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

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
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

class UIEPretrainConstraintSeq2SeqTrainer(OriginalConstraintSeq2SeqTrainer):

    def pretrain(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> nn.Module:
        def get_loss(model, inputs):
            if is_sagemaker_mp_enabled():
                scaler = self.scaler if self.use_amp else None
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps
            
            return loss

        model.train()
        inputs = self._prepare_inputs(inputs)

        # record inputs
        record_input_ids = inputs.pop("record_input_ids")
        record_inputs = {
            "input_ids": record_input_ids,
            "labels": inputs["labels"],
            "decoder_input_ids": inputs["decoder_input_ids"]
        }

        # mlm inputs
        mlm_input_ids = inputs.pop("mlm_input_ids")
        mlm_target_ids = inputs.pop("mlm_target_ids")
        mlm_decoder_input_ids = inputs.pop("mlm_decoder_input_ids")
        mlm_inputs = {
            "input_ids": mlm_input_ids,
            "labels": mlm_target_ids,
            "decoder_input_ids": mlm_decoder_input_ids,
        }

        # remove other field
        if "noised_input_ids" in inputs.keys():
            inputs.pop("noised_input_ids")
        if "noised_att_mask" in inputs.keys():
            inputs.pop("noised_att_mask")

        # inner loop
        loss = get_loss(model, inputs)
        
        # record loss
        record_loss = get_loss(model, record_inputs)

        # mlm loss
        mlm_loss = get_loss(model, mlm_inputs)

        loss = loss + record_loss + mlm_loss

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
        return loss.detach()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        oom = False
        oom_message = ""
        try:
            loss = self.pretrain(model, inputs)
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
                oom_message = str(e)
                logger.warning(f'ran out of memory {self.oom_batch} on {self.args.local_rank}')
                for k, v in inputs.items():
                    print(k, v.size())
            else:
                raise e

        if oom:
            self.oom_batch += 1
            raise RuntimeError(oom_message)

class UIEFinetuneConstraintSeq2SeqTrainer(OriginalConstraintSeq2SeqTrainer):

    def finetune(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> nn.Module:
        def get_loss(model, inputs):
            if is_sagemaker_mp_enabled():
                scaler = self.scaler if self.use_amp else None
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps
            
            return loss

        model.train()
        inputs = self._prepare_inputs(inputs)

        # remove pretrain field
        if "record_input_ids" in inputs.keys():
            inputs.pop("record_input_ids")
        if "mlm_input_ids" in inputs.keys():
            inputs.pop("mlm_input_ids")
        if "mlm_target_ids" in inputs.keys():
            inputs.pop("mlm_target_ids")
        if "mlm_decoder_input_ids" in inputs.keys():
            inputs.pop("mlm_decoder_input_ids")
        if "noised_input_ids" in inputs.keys():
            inputs.pop("noised_input_ids")
        if "noised_att_mask" in inputs.keys():
            inputs.pop("noised_att_mask")


        # finetune loss
        loss = get_loss(model, inputs)

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
        return loss.detach()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        oom = False
        oom_message = ""
        try:
            loss = self.finetune(model, inputs)
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
                oom_message = str(e)
                logger.warning(f'ran out of memory {self.oom_batch} on {self.args.local_rank}')
                for k, v in inputs.items():
                    print(k, v.size())
            else:
                raise e

        if oom:
            self.oom_batch += 1
            raise RuntimeError(oom_message)

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

        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # remove pretrain field
        if "record_input_ids" in inputs.keys():
            inputs.pop("record_input_ids")
        if "mlm_input_ids" in inputs.keys():
            inputs.pop("mlm_input_ids")
        if "mlm_target_ids" in inputs.keys():
            inputs.pop("mlm_target_ids")
        if "mlm_decoder_input_ids" in inputs.keys():
            inputs.pop("mlm_decoder_input_ids")
        if "noised_input_ids" in inputs.keys():
            inputs.pop("noised_input_ids")
        if "noised_att_mask" in inputs.keys():
            inputs.pop("noised_att_mask")

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

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
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

class MetaPretrainConstraintSeq2SeqTrainer(OriginalConstraintSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = l2l.algorithms.MAML(self.model, lr=1e-4, first_order=True)
        self.model_wrapped = self.model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = unwrap_model(self.model).state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def cat_two_various_length_tensor(self, x, y, max_length=None):
        batch_size, seq_len = x.shape

        def create_index(x, y, batch_size):
            num = torch.bincount(((x!=0)&(x!=1)).nonzero()[:,0], minlength=batch_size)
            
            pad_ones = torch.ones(y.shape[0], y.shape[1]-1, dtype=torch.long, device=y.device)
            
            update_index = torch.cat([num.unsqueeze(-1), pad_ones], dim=-1)
            index = torch.cumsum(update_index, dim=1)
            
            return index
        
        xy = torch.cat([x, torch.zeros_like(y)], dim=-1)
        index = create_index(x, y, batch_size)

        xy.scatter_(1, index, y)
        
        if max_length is not None:
            xy = xy[:, :max_length]
        else:
            xy = xy[:, :seq_len]
        
        att_mask = (xy != 0).float()

        return xy, att_mask
    
    def split_input(self, inputs):
        support = {}
        query = {}
        for key, value in inputs.items():
            length = value.shape[0]
            # support[key] = value[:length//2]
            # query[key] = value[length//2:]
            support[key] = value[::2].contiguous()
            query[key] = value[1::2].contiguous()
        return support, query

    def meta_learn(self, meta_learner: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> nn.Module:
        def get_loss(model, inputs):
            if is_sagemaker_mp_enabled():
                scaler = self.scaler if self.use_amp else None
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps
            
            return loss
        
        model = meta_learner.clone()

        model.train()
        inputs = self._prepare_inputs(inputs)

        #########################################################
        # mlm loss and record loss
        #########################################################

        # record inputs
        record_input_ids = inputs.pop("record_input_ids")
        record_inputs = {
            "input_ids": record_input_ids,
            "labels": inputs["labels"],
            "decoder_input_ids": inputs["decoder_input_ids"]
        }

        # mlm inputs
        mlm_input_ids = inputs.pop("mlm_input_ids")
        mlm_target_ids = inputs.pop("mlm_target_ids")
        mlm_decoder_input_ids = inputs.pop("mlm_decoder_input_ids")
        mlm_inputs = {
            "input_ids": mlm_input_ids,
            "labels": mlm_target_ids,
            "decoder_input_ids": mlm_decoder_input_ids,
        }

        # record loss
        record_loss = get_loss(model, record_inputs)

        # mlm loss
        mlm_loss = get_loss(model, mlm_inputs)

        #########################################################
        # meta loss
        #########################################################

        support_inputs, query_inputs = self.split_input(inputs)

        # refine inputs
        support_refine_input_ids = support_inputs.pop("noised_input_ids")
        support_refine_att_mask = support_inputs.pop("noised_att_mask")
        support_refine_inputs = {
            "input_ids": support_refine_input_ids,
            "attention_mask": support_refine_att_mask,
            "labels": support_inputs["labels"],
            "decoder_input_ids": support_inputs["decoder_input_ids"]
        }

        query_refine_input_ids = query_inputs.pop("noised_input_ids")
        query_refine_att_mask = query_inputs.pop("noised_att_mask")
        query_refine_inputs = {
            "input_ids": query_refine_input_ids,
            "attention_mask": query_refine_att_mask,
            "labels": query_inputs["labels"],
            "decoder_input_ids": query_inputs["decoder_input_ids"]
        }

        # inner loss
        support_text2struct_loss = get_loss(model, support_inputs)
        support_refine_loss = get_loss(model, support_refine_inputs)
        support_loss = support_text2struct_loss + support_refine_loss
        
        model.adapt(support_loss)

        # outer loss
        # support_text2struct_loss = get_loss(model, support_inputs)
        # support_refine_loss = get_loss(model, support_refine_inputs)
        # support_loss = support_text2struct_loss + support_refine_loss

        query_text2struct_loss = get_loss(model, query_inputs)
        query_refine_loss = get_loss(model, query_refine_inputs)
        query_loss = query_text2struct_loss + query_refine_loss

        # meta_loss = (query_loss + support_loss)
        meta_loss = query_loss

        loss = (meta_loss + record_loss + mlm_loss) / 4

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
        return loss.detach()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        oom = False
        oom_message = ""
        try:
            loss = self.meta_learn(model, inputs)
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
                oom_message = str(e)
                logger.warning(f'ran out of memory {self.oom_batch} on {self.args.local_rank}')
                for k, v in inputs.items():
                    print(k, v.size())
            else:
                raise e

        if oom:
            self.oom_batch += 1
            raise RuntimeError(oom_message)

    def prediction_in_one_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        def prefix_allowed_tokens_fn(batch_id, sent):
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if hasattr(self, "_max_length") and self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if hasattr(self, "_num_beams") and self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

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
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        loss, generated_tokens, labels = self.prediction_in_one_step(model, inputs, prediction_loss_only, ignore_keys)

        original_input_ids = inputs["input_ids"]

        new_input_ids, new_att_mask = self.cat_two_various_length_tensor(original_input_ids, generated_tokens)

        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_att_mask

        loss, generated_tokens, labels = self.prediction_in_one_step(model, inputs, prediction_loss_only, ignore_keys)

        return loss, generated_tokens, labels

class MetaFinetuneConstraintSeq2SeqTrainer(OriginalConstraintSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_text = " Predicted results: "
        self.prompt_inputs = self.tokenizer(self.prompt_text, return_tensors="pt")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = unwrap_model(self.model).state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def cat_prompt(self, original_input_ids):
        prompt = self.prompt_inputs["input_ids"]
        prompt = prompt.to(original_input_ids.device)

        batch_size = original_input_ids.shape[0]
        prompt = prompt.repeat(batch_size, 1)

        prompt_len = prompt.shape[1]
        text_len = original_input_ids.shape[1]

        new_input_ids, new_att_mask = self.cat_two_various_length_tensor(original_input_ids, prompt, max_length=text_len+prompt_len)
        return new_input_ids, new_att_mask

    def cat_two_various_length_tensor(self, x, y, max_length=None):
        batch_size, seq_len = x.shape

        def create_index(x, y, batch_size):
            num = torch.bincount(((x!=0)&(x!=1)).nonzero()[:,0], minlength=batch_size)
            
            pad_ones = torch.ones(y.shape[0], y.shape[1]-1, dtype=torch.long, device=y.device)
            
            update_index = torch.cat([num.unsqueeze(-1), pad_ones], dim=-1)
            index = torch.cumsum(update_index, dim=1)
            
            return index
        
        xy = torch.cat([x, torch.zeros_like(y)], dim=-1)
        index = create_index(x, y, batch_size)

        xy.scatter_(1, index, y)

        if max_length is not None:
            xy = xy[:, :max_length]
        else:
            xy = xy[:, :seq_len]

        att_mask = (xy != 0).float()

        return xy, att_mask
    
    def finetune(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> nn.Module:
        def get_loss(model, inputs):
            if is_sagemaker_mp_enabled():
                scaler = self.scaler if self.use_amp else None
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps
            
            return loss

        model.train()
        inputs = self._prepare_inputs(inputs)

        # remove pretrain field
        if "record_input_ids" in inputs.keys():
            inputs.pop("record_input_ids")
        if "mlm_input_ids" in inputs.keys():
            inputs.pop("mlm_input_ids")
        if "mlm_target_ids" in inputs.keys():
            inputs.pop("mlm_target_ids")
        if "mlm_decoder_input_ids" in inputs.keys():
            inputs.pop("mlm_decoder_input_ids")
        if "noised_input_ids" in inputs.keys():
            inputs.pop("noised_input_ids")
        if "noised_att_mask" in inputs.keys():
            inputs.pop("noised_att_mask")

        # store original input_ids
        original_input_ids = inputs["input_ids"]
        original_att_mask = inputs["attention_mask"]

        # finetune loss
        loss = get_loss(model, inputs)

        # refine loss
        _, generated_tokens, labels = self.prediction_in_one_step(model, inputs, prediction_loss_only=False)

        input_ids_with_refine_prompt, att_mask_with_refine_prompt = self.cat_prompt(original_input_ids)
        new_input_ids, new_att_mask = self.cat_two_various_length_tensor(input_ids_with_refine_prompt, generated_tokens)
        # new_input_ids, new_att_mask = self.cat_two_various_length_tensor(original_input_ids, generated_tokens)
        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_att_mask

        # refine loss
        refine_loss = get_loss(model, inputs)

        loss = loss + refine_loss
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
        return loss.detach()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        oom = False
        oom_message = ""
        try:
            loss = self.finetune(model, inputs)
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
                oom_message = str(e)
                logger.warning(f'ran out of memory {self.oom_batch} on {self.args.local_rank}')
                for k, v in inputs.items():
                    print(k, v.size())
            else:
                raise e

        if oom:
            self.oom_batch += 1
            raise RuntimeError(oom_message)

    def prediction_in_one_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        def prefix_allowed_tokens_fn(batch_id, sent):
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # remove pretrain field
        if "record_input_ids" in inputs.keys():
            inputs.pop("record_input_ids")
        if "mlm_input_ids" in inputs.keys():
            inputs.pop("mlm_input_ids")
        if "mlm_target_ids" in inputs.keys():
            inputs.pop("mlm_target_ids")
        if "mlm_decoder_input_ids" in inputs.keys():
            inputs.pop("mlm_decoder_input_ids")
        if "noised_input_ids" in inputs.keys():
            inputs.pop("noised_input_ids")
        if "noised_att_mask" in inputs.keys():
            inputs.pop("noised_att_mask")

        gen_kwargs = {
            "max_length": self._max_length if hasattr(self, "_max_length") and self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if hasattr(self, "_num_beams") and self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

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
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        # store original input_ids
        original_input_ids = inputs["input_ids"]
        original_att_mask = inputs["attention_mask"]

        # first prediction
        loss, generated_tokens, labels = self.prediction_in_one_step(model, inputs, prediction_loss_only, ignore_keys)

        # refine
        input_ids_with_refine_prompt, att_mask_with_refine_prompt = self.cat_prompt(original_input_ids)
        new_input_ids, new_att_mask = self.cat_two_various_length_tensor(input_ids_with_refine_prompt, generated_tokens)
        # new_input_ids, new_att_mask = self.cat_two_various_length_tensor(original_input_ids, generated_tokens)
        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_att_mask

        loss, generated_tokens, labels = self.prediction_in_one_step(model, inputs, prediction_loss_only, ignore_keys)

        return loss, generated_tokens, labels

###########################################################################

ConstraintSeq2SeqTrainer = OriginalConstraintSeq2SeqTrainer

def main(): pass


if __name__ == "__main__":
    main()
