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

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomIOPOTrainer(DPOTrainer):
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
        self.reference_free = False ######## Modify
        # self.reference_free = True
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

        # iopo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.iopo_label_smoothing
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

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

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
    
    def simpo_loss_v1(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", rejected_logps_v1: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = 2*chosen_logps - rejected_logps - 0.5*rejected_logps_v1 
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss
    
    def iopo_loss(
        self,
        policy_chosen_logps_v1: torch.FloatTensor,
        policy_rejected_logps_v1: torch.FloatTensor,
        policy_chosen_logps_v2: torch.FloatTensor,
        policy_rejected_logps_v2: torch.FloatTensor,
        reference_chosen_logps_v1: torch.FloatTensor,
        reference_rejected_logps_v1: torch.FloatTensor,
        reference_chosen_logps_v2: torch.FloatTensor,
        reference_rejected_logps_v2: torch.FloatTensor,
        chosen_length_v1, 
        rejected_length_v1,
        chosen_length_v2,
        rejected_length_v2
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the IOPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the IOPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios_v1 = policy_chosen_logps_v1.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps_v1.to(self.accelerator.device)
        rejected_logratios_v1 = policy_rejected_logps_v1.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps_v1.to(self.accelerator.device)

        # if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
        #     # The alpha-divergence formula: (1 - u^-alpha) / alpha
        #     # The divergence difference between the chosen and rejected sample is:
        #     #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
        #     #        = (u[l]^-alpha - u[w]^-alpha) / alpha
        #     # where u[w] and u[l] are the policy/reference probability ratios
        #     # for the chosen and rejected samples, respectively.
        #     alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
        #     if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
        #         alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
        #     logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        # else:

        pi_logratios_1 = 2*policy_chosen_logps_v1 - policy_rejected_logps_v1 - policy_rejected_logps_v2
        pi_logratios_2 = 2*policy_chosen_logps_v2 - policy_rejected_logps_v2 - policy_rejected_logps_v1
        # pi_logratios = policy_chosen_logps - policy_rejected_logps + 0.1*(policy_chosen_logps - policy_rejected_logps_v1) ######
        if self.reference_free:
            ref_logratios_1 = torch.tensor([0], dtype=pi_logratios_1.dtype, device=pi_logratios_1.device)
            ref_logratios_2 = torch.tensor([0], dtype=pi_logratios_2.dtype, device=pi_logratios_2.device)
        else:
            ref_logratios_1 = 2*reference_chosen_logps_v1 - reference_rejected_logps_v1 - reference_rejected_logps_v2 #####
            ref_logratios_2 = 2*reference_chosen_logps_v2 - reference_rejected_logps_v2 - reference_rejected_logps_v1 #####

        pi_logratios_1 = pi_logratios_1.to(self.accelerator.device)
        ref_logratios_1 = ref_logratios_1.to(self.accelerator.device)
        pi_logratios_2 = pi_logratios_2.to(self.accelerator.device)
        ref_logratios_2 = ref_logratios_2.to(self.accelerator.device)
        logits = 0.5* (pi_logratios_1 - ref_logratios_1 + pi_logratios_2 - ref_logratios_2)
        ###
        # sft_loss = -policy_chosen_logps/chosen_length #####
        # sft_loss = -policy_chosen_logps #####
        ###
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # logits = pi_logratios - ref_logratios
        # penalty_term = torch.maximum(torch.zeros_like(policy_chosen_logps), reference_chosen_logps - policy_chosen_logps) ######
        # logits += - 50 * penalty_term  ######

        # losses = (
        #         -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        #         - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        #     )
        ###

            # if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
            #     # The js-divergence formula: log(2 * u / (1 + u))
            #     # The divergence difference between the chosen and rejected sample is:
            #     #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
            #     #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
            #     # where u[w] and u[l] are the policy/reference probability ratios
            #     # for the chosen and rejected samples, respectively.
            #     logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the IOPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative IOPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            # losses = sft_loss + losses #####
            # losses = sft_loss
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://arxiv.org/pdf/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios_sorted, _ = torch.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)

            delta = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust', 'exo_pair']"
            )

        chosen_rewards_v1 = (
            self.beta
            * (
                policy_chosen_logps_v1.to(self.accelerator.device) - reference_chosen_logps_v1.to(self.accelerator.device)
            ).detach()
        )
        chosen_rewards_v2 = (
            self.beta
            * (
                policy_chosen_logps_v2.to(self.accelerator.device) - reference_chosen_logps_v2.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards_v1 = (
            self.beta
            * (
                policy_rejected_logps_v1.to(self.accelerator.device)
                - reference_rejected_logps_v1.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards_v2 = (
            self.beta
            * (
                policy_rejected_logps_v2.to(self.accelerator.device)
                - reference_rejected_logps_v2.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards_v1, rejected_rewards_v1, chosen_rewards_v2, rejected_rewards_v2

    def compute_preference_loss(
        self,
        policy_chosen_logps_v1: "torch.Tensor",
        policy_rejected_logps_v1: "torch.Tensor",
        policy_chosen_logps_v2: "torch.Tensor",
        policy_rejected_logps_v2: "torch.Tensor",
        reference_chosen_logps_v1: Optional["torch.Tensor"],
        reference_rejected_logps_v1: Optional["torch.Tensor"],
        reference_chosen_logps_v2: Optional["torch.Tensor"],
        reference_rejected_logps_v2: Optional["torch.Tensor"],
        chosen_length_v1, 
        rejected_length_v1,
        chosen_length_v2, 
        rejected_length_v2,
        
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        # if not self.finetuning_args.use_ref_model:
        #     if self.loss_type == "orpo":
        #         losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        #     elif self.loss_type == "simpo":
        #         losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        #     else:
        #         raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

        #     chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        #     rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        # else:
        losses, chosen_rewards_v1, rejected_rewards_v1, chosen_rewards_v2, rejected_rewards_v2 = self.iopo_loss(
            policy_chosen_logps_v1, policy_rejected_logps_v1, policy_chosen_logps_v2, policy_rejected_logps_v2, reference_chosen_logps_v1, reference_rejected_logps_v1, reference_chosen_logps_v2, reference_rejected_logps_v2, chosen_length_v1, rejected_length_v1, chosen_length_v2, rejected_length_v2
        ) ####
            ######
            # losses = self.simpo_loss_v1(policy_chosen_logps, policy_rejected_logps, policy_rejected_logps_v1)
            # chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            # rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
            # rejected_rewards_v1 = self.beta * policy_rejected_logps_v1.to(self.accelerator.device).detach()
            # sft_loss = losses
            ######

        return losses, chosen_rewards_v1, rejected_rewards_v1, chosen_rewards_v2, rejected_rewards_v2

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)

        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]: #########sigmoid注意适当移除########0828
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 4
        chosen_logps_v1, rejected_logps_v1, chosen_logps_v2, rejected_logps_v2 = all_logps.split(batch_size, dim=0)
        chosen_logits_v1, rejected_logits_v1, chosen_logits_v2, rejected_logits_v2 = all_logits.split(batch_size, dim=0)
        # chosen_length, _ = valid_length.split(batch_size, dim=0) ###
        chosen_length_v1, rejected_length_v1, chosen_length_v2, rejected_length_v2 = valid_length.split(batch_size, dim=0)
        return chosen_logps_v1, rejected_logps_v1, chosen_logps_v2, rejected_logps_v2, chosen_logits_v1, rejected_logits_v1, chosen_logits_v2, rejected_logits_v2, chosen_logps_v1 / chosen_length_v1, chosen_length_v1, rejected_length_v1, chosen_logps_v2 / chosen_length_v2, chosen_length_v2, rejected_length_v2

    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
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
            reference_chosen_logps_v1, reference_rejected_logps_v1, reference_chosen_logps_v2, reference_rejected_logps_v2, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps_v1, reference_rejected_logps_v1, reference_chosen_logps_v2, reference_rejected_logps_v2

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the IOPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps_v1,
            policy_rejected_logps_v1,
            policy_chosen_logps_v2,
            policy_rejected_logps_v2,
            policy_chosen_logits_v1,
            policy_rejected_logits_v1,
            policy_chosen_logits_v2,
            policy_rejected_logits_v2,
            policy_chosen_logps_avg_v1,
            chosen_length_v1, rejected_length_v1, 
            policy_chosen_logps_avg_v2, 
            chosen_length_v2, rejected_length_v2
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps_v1, reference_rejected_logps_v1, reference_chosen_logps_v2, reference_rejected_logps_v2 = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards_v1, rejected_rewards_v1, chosen_rewards_v2, rejected_rewards_v2 = self.compute_preference_loss(
            policy_chosen_logps_v1,
            policy_rejected_logps_v1,
            policy_chosen_logps_v2,
            policy_rejected_logps_v2,
            reference_chosen_logps_v1,
            reference_rejected_logps_v1,
            reference_chosen_logps_v2,
            reference_rejected_logps_v2,
            chosen_length_v1, rejected_length_v1, chosen_length_v2, rejected_length_v2
        ) #####
        sft_loss = -policy_chosen_logps_avg_v1 - policy_chosen_logps_avg_v2
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        reward_accuracies_v1 = (chosen_rewards_v1 > rejected_rewards_v1).float()
        reward_accuracies_v2 = (chosen_rewards_v2 > rejected_rewards_v2).float()

        prefix = "eval_" if train_eval == "eval" else ""
        # metrics["{}sft_loss".format(prefix)] = sft_loss_iopo.detach().mean().cpu() #####
        metrics["{}loss".format(prefix)] = losses.detach().mean().cpu() #####
        metrics["{}rewards/chosen_v1".format(prefix)] = chosen_rewards_v1.mean().cpu()
        metrics["{}rewards/rejected_v1".format(prefix)] = rejected_rewards_v1.mean().cpu()
        metrics["{}rewards/chosen_v2".format(prefix)] = chosen_rewards_v2.mean().cpu()
        metrics["{}rewards/rejected_v2".format(prefix)] = rejected_rewards_v2.mean().cpu()
        metrics["{}rewards/accuracies_v1".format(prefix)] = reward_accuracies_v1.mean().cpu()
        metrics["{}rewards/accuracies_v2".format(prefix)] = reward_accuracies_v2.mean().cpu()
        metrics["{}rewards/margins_v1".format(prefix)] = (chosen_rewards_v1 - rejected_rewards_v1).mean().cpu()
        metrics["{}rewards/margins_v2".format(prefix)] = (chosen_rewards_v2 - rejected_rewards_v2).mean().cpu()
        metrics["{}logps/rejected_v1".format(prefix)] = policy_rejected_logps_v1.detach().mean().cpu()
        metrics["{}logps/rejected_v2".format(prefix)] = policy_rejected_logps_v2.detach().mean().cpu()
        metrics["{}logps/chosen_v1".format(prefix)] = policy_chosen_logps_v1.detach().mean().cpu()
        metrics["{}logps/chosen_v2".format(prefix)] = policy_chosen_logps_v2.detach().mean().cpu()
        metrics["{}logps/ref_rejected_v1".format(prefix)] = reference_rejected_logps_v1.detach().mean().cpu() ###
        metrics["{}logps/ref_rejected_v2".format(prefix)] = reference_rejected_logps_v2.detach().mean().cpu() ###
        metrics["{}logps/ref_chosen_v1".format(prefix)] = reference_chosen_logps_v1.detach().mean().cpu() ###
        metrics["{}logps/ref_chosen_v2".format(prefix)] = reference_chosen_logps_v2.detach().mean().cpu() ###
        metrics["{}logits/rejected_v1".format(prefix)] = policy_rejected_logits_v1.detach().mean().cpu()
        metrics["{}logits/rejected_v2".format(prefix)] = policy_rejected_logits_v2.detach().mean().cpu()
        metrics["{}logits/chosen_v1".format(prefix)] = policy_chosen_logits_v1.detach().mean().cpu()
        metrics["{}logits/chosen_v2".format(prefix)] = policy_chosen_logits_v2.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics
