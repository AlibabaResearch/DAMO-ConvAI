# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    def count_non_neg100_segments(self, tensor):
        """
        计算每个子tensor中不为-100的连续子段数量。
        
        Args:
            tensor (torch.Tensor): 输入形状为 (batch_size, 4096) 的张量。
        
        Returns:
            torch.Tensor: 形状为 (batch_size,) 的张量，每个元素对应每个输入行中的连续子段数量。
        """
        # 创建掩码，True表示不为-100
        mask = tensor != -100  # shape: (batch_size, 4096)
        
        # 对掩码进行偏移，计算变化点
        shifted_mask = torch.zeros_like(mask)
        shifted_mask[:, 1:] = mask[:, :-1]
        
        # 找到从False到True的地方
        transitions = (~shifted_mask) & mask  # shape: (batch_size, 4096)
        
        # 统计每行中的True数量，即转换次数
        counts = transitions.sum(dim=1)
        
        # 如果第一位也是不为-100，则需要加1
        # first_non_neg100 = mask[:, 0].long()
        # counts += first_non_neg100
        
        return counts

    def get_batch_logps_DMPO(
        self, logits: "torch.Tensor", labels: "torch.Tensor", num_segments, label_pad_token_id: int = IGNORE_INDEX
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the log probabilities of the given labels under the given logits.

        Returns:
            logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
            valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != label_pad_token_id).float()
        labels[labels == label_pad_token_id] = 0  # dummy token
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        for i in range(loss_mask.shape[0]):
            cnt = 0
            for j in range(loss_mask.shape[1]):
                if loss_mask[i, j] == 1:
                    loss_mask[i, j] *= (1 - 0.7**(num_segments[i]-cnt)) / (1 - 0.7**(num_segments[i]))
                if j >= 1 and loss_mask[i, j-1] != 0 and loss_mask[i, j] == 0:
                    cnt += 1
            try:
                assert cnt == num_segments[i]-1 or cnt == num_segments[i], loss_mask[i]
            except:
                print(num_segments[i])
                print(labels[i, -30:])
                exit()
        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def get_batch_logps_gama(
        self, logits: "torch.Tensor", labels: "torch.Tensor", num_segments, label_pad_token_id: int = IGNORE_INDEX
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the log probabilities of the given labels under the given logits.

        Returns:
            logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
            valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != label_pad_token_id).float()
        labels[labels == label_pad_token_id] = 0  # dummy token
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        gama = 0.8

        for i in range(loss_mask.shape[0]):
            cnt = 0
            for j in range(loss_mask.shape[1]):
                if loss_mask[i, j] == 1:
                    loss_mask[i, j] *= gama**(num_segments[i]-cnt)
                if j >= 1 and loss_mask[i, j-1] != 0 and loss_mask[i, j] == 0:
                    cnt += 1
            try:
                assert cnt == num_segments[i]-1 or cnt == num_segments[i], loss_mask[i]
            except:
                print(num_segments[i])
                print(labels[i, -30:])
                exit()
        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def get_batch_logps_for_dpo(
        self, logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the log probabilities of the given labels under the given logits.

        Returns:
            logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
            valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    cnt = j
                    while cnt < labels.shape[1] and labels[i, cnt] != -100:
                        cnt += 1
                    labels[i, cnt:] = -100

        logits = logits[:, :-1, :]
        loss_mask = (labels != label_pad_token_id).float()
        labels[labels == label_pad_token_id] = 0  # dummy token
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


    # @override
    # def concatenated_forward(
    #     self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    # ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    #     r"""
    #     Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

    #     Otherwise the average log probabilities.
    #     """
    #     if self.finetuning_args.use_ref_model:
    #         batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

    #     all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
    #     all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
    #     if self.loss_type in ["ipo", "orpo", "simpo"]:
    #         all_logps = all_logps / valid_length

    #     # turn-level average
    #     # num_segments = self.count_non_neg100_segments(batch["labels"])
    #     # all_logps /= num_segments

    #     # DMPO
    #     # num_segments = self.count_non_neg100_segments(batch["labels"])
    #     # all_logps, _ = self.get_batch_logps_DMPO(all_logits, batch["labels"], num_segments)

    #     # DPO
    #     all_logps_dpo, _ = self.get_batch_logps_for_dpo(all_logits, batch["labels"])

    #     batch_size = batch["input_ids"].size(0) // 2

    #     chosen_logps_all, rejected_logps_all = all_logps.split(batch_size, dim=0)
    #     # chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)

    #     chosen_logps, rejected_logps = all_logps_dpo.split(batch_size, dim=0)
    #     # chosen_logits, rejected_logits = all_logits_dpo.split(batch_size, dim=0)

    #     chosen_length, _ = valid_length.split(batch_size, dim=0)
    #     return chosen_logps, rejected_logps, chosen_logps_all, rejected_logps_all, chosen_logps / chosen_length

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        # turn-level average
        # num_segments = self.count_non_neg100_segments(batch["labels"])
        # all_logps /= num_segments

        # DMPO
        # num_segments = self.count_non_neg100_segments(batch["labels"])
        # all_logps, _ = self.get_batch_logps_DMPO(all_logits, batch["labels"], num_segments)

        # gama
        # num_segments = self.count_non_neg100_segments(batch["labels"])
        # all_logps, _ = self.get_batch_logps_gama(all_logits, batch["labels"], num_segments)

        batch_size = batch["input_ids"].size(0) // 2

        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logps/ref_chosen".format(prefix)] = reference_chosen_logps.detach().mean().cpu()
        metrics["{}logps/ref_rejected".format(prefix)] = reference_rejected_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics
