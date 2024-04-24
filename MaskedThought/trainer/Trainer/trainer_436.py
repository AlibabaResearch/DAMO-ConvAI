
import os
from dataclasses import dataclass
import numpy as np
from numpy.lib.function_base import average, median
import torch
from torch.distributed.distributed_c10d import init_process_group
from transformers import Trainer as BaseTrainer
# from trainer.Trainer.trainer import Trainer as BaseTrainer
# from trainer.Trainer.trainer import MetricsCalculator, AmlLogger
from data.data_reader import CustomizeCollator
import trainer.Metrics as Metrics
import trainer.Outputter as Outputter
from transformers.modeling_utils import unwrap_model
from transformers import logging

try:
    from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
except:
    pass
# try:
#     from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
# except:
#     assert 1==0
#     def is_deepspeed_available():
#         return False
from packaging import version
from transformers.trainer_callback import PrinterCallback,TrainerCallback, TrainerState
from config.decorator import replace
from torch.utils.data.dataloader import DataLoader
import copy
from torch.utils.data.dataset import Dataset, IterableDataset
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard, LabelSmoother
from transformers.trainer_utils import TrainOutput, has_length, speed_metrics
import math
import nltk
import torch.nn.functional as F
import random
try:
    from peft import PeftModelForSeq2SeqLM,  PeftModelForCausalLM
except:
    pass
import time

from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    get_model_param_count,
)



from data.tokenizer_utils import *
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math 
import torch.distributed as dist

from transformers import get_linear_schedule_with_warmup

from trainer.Trainer import register_trainer
from collections import deque
import collections
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.utils.data.distributed import DistributedSampler

if is_datasets_available():
    import datasets

import logging as local_logging
logger = logging.get_logger(__name__)
logger.setLevel('INFO')
local_logging.basicConfig(format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",level=logging.INFO)


from transformers.integrations import AzureMLCallback

from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

class HisStatisticInfo:
    def __init__(self, args):
        self.args = args
        self.his_dif_cont_rate = []

        self.his_ce_loss = []
        self.his_loss = []


        self.his_2_gram_loss = []
        self.his_3_gram_loss = []
        self.his_4_gram_loss = []
        self.his_2_gram_acc = []
        self.his_3_gram_acc = []
        self.his_4_gram_acc = []

        self.his_1_gram_ent = []
        self.his_2_gram_ent = []
        self.his_3_gram_ent = []
        self.his_4_gram_ent = []

        self.his_static_r = []

        self.his_forward_time = []

        self.his_backward_time = []
        self.mask_rate = 0
        self.print_every = args.print_every
    def update(self):

        self.his_ce_loss = self.his_ce_loss[-self.print_every:]
        self.his_loss = self.his_loss[-self.print_every:]
        self.his_2_gram_acc = self.his_2_gram_acc[-self.print_every:]
        self.his_3_gram_acc = self.his_3_gram_acc[-self.print_every:]
        self.his_4_gram_acc = self.his_4_gram_acc[-self.print_every:]
        self.his_2_gram_loss = self.his_2_gram_loss[-self.print_every:]
        self.his_3_gram_loss = self.his_3_gram_loss[-self.print_every:]
        self.his_4_gram_loss = self.his_4_gram_loss[-self.print_every:]

        self.his_1_gram_ent = self.his_1_gram_ent[-self.print_every:]
        self.his_2_gram_ent = self.his_2_gram_ent[-self.print_every:]
        self.his_3_gram_ent = self.his_3_gram_ent[-self.print_every:]
        self.his_4_gram_ent = self.his_4_gram_ent[-self.print_every:]
        self.his_dif_cont_rate = self.his_dif_cont_rate[-self.print_every:]
        self.his_static_r = self.his_static_r[-self.print_every:]

    def print_info(self, tokenizer, step):

        logger.info('At step {}, his_ce_loss {}'.format(step, np.mean(self.his_ce_loss)))

        logger.info('At step {}, his_loss {}'.format(step, np.mean(self.his_loss)))
        logger.info('At step {}, his_dif_cont_rate'.format(step, np.mean(self.his_dif_cont_rate)))
        logger.info('At step {}, gram_acc: 2:{}, 3:{}, 4:{}'.format(step, np.mean(self.his_2_gram_acc),
                                                                    np.mean(self.his_3_gram_acc),
                                                                    np.mean(self.his_4_gram_acc)))
        logger.info('At step {}, gram_loss: 2:{}, 3:{}, 4:{}'.format(step, np.mean(self.his_2_gram_loss),
                                                                    np.mean(self.his_3_gram_loss),
                                                                    np.mean(self.his_4_gram_loss)))
        logger.info('At step {}, gram_ent: 1: {}, 2:{}, 3:{}, 4:{}'.format(step,np.mean(self.his_1_gram_ent), np.mean(self.his_2_gram_ent),
                                                                     np.mean(self.his_3_gram_ent),
                                                                     np.mean(self.his_4_gram_ent)))
        logger.info('At step {}, his_forward_time {}'.format(step, np.mean(self.his_forward_time)))
        logger.info('At step {}, his_backward_time {}'.format(step, np.mean(self.his_backward_time)))

        logger.info('At step {}, mask_rate {}'.format(step, np.mean(self.mask_rate)))
        logger.info('At step {}, lr {}'.format(step, self.lr))




@register_trainer("trainer436")
class Trainer(BaseTrainer):
    def __init__(self, model, args, model_args, task_args, train_dataset, eval_dataset, auto_tokenizer):

        data_collator = CustomizeCollator(train_dataset, eval_dataset)
        metrics_calculator = None
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset.get_dataset() if train_dataset else None,
            eval_dataset=eval_dataset.get_dataset() if eval_dataset else None,
            data_collator=data_collator,
            compute_metrics=metrics_calculator
        )
        self.args = args
        self.tokenizer = auto_tokenizer
        auto_tokenizer.truncation_side = "right"

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            if isinstance(model.module, PeftModelForCausalLM) or isinstance(model.module, PeftModelForSeq2SeqLM):
                self.model.module.base_model.model.tokenizer = auto_tokenizer
            else:
                self.model.module.tokenizer = auto_tokenizer
        else:
            try:
                if isinstance(model, PeftModelForCausalLM) or isinstance(model, PeftModelForSeq2SeqLM):
                    self.model.base_model.model.tokenizer = auto_tokenizer
                else:
                    self.model.tokenizer = auto_tokenizer
            except:
                self.model.tokenizer = auto_tokenizer


        self.his_info = HisStatisticInfo(args)

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.max_steps = max_steps
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                self.his_info.his_ce_loss.append(tr_loss_step.float().clone().detach().cpu().numpy() * args.gradient_accumulation_steps)
                self.his_info.his_loss.append(tr_loss_step.float().clone().detach().cpu().numpy() * args.gradient_accumulation_steps)

                print_every = args.print_every

                output_dir = args.output_dir

                # if not args.not_save:
                #     if args.zero3:
                #         print_every = args.print_every
                #         save_every = args.save_every
                #         output_dir = args.output_dir
                #
                #         if (step + 1) % save_every == 0:
                #             if args.local_rank <= 0:
                #                 if not os.path.exists(output_dir + '/' + args.exp_name):
                #                     os.makedirs(output_dir + '/' + args.exp_name)
                #             save_path = output_dir + '/' + args.exp_name + '/_model.{}_{}'.format(self.state.global_step,
                #                                                                                   step)
                #             # logger.info('save to {}, epoch: {}'.format(save_path, self.state.epoch))
                #             logger.info('save to {}, epoch: {}'.format(save_path + '_adapter', self.state.epoch))
                #             self.accelerator.wait_for_everyone()
                #             if self.accelerator.is_main_process:
                #                 self.tokenizer.save_pretrained(save_path + '_adapter')
                #             unwrapped_model = self.accelerator.unwrap_model(model)
                #             state_dict = self.accelerator.get_state_dict(model)
                #             unwrapped_model.save_pretrained(
                #                 save_path + '_adapter', is_main_process=self.accelerator.is_main_process,
                #                 save_function=self.accelerator.save, state_dict=state_dict
                #             )
                #         if args.local_rank <= 0:
                #             if (step + 1) % print_every == 0:
                #                 self.his_info.lr = self.optimizer.param_groups[0]['lr']
                #                 self.his_info.update()
                #                 self.his_info.print_info(tokenizer=self.tokenizer, step=step)
                #     else:
                #
                #         if (step + 1) % save_every == 0:
                #             if args.local_rank <= 0:
                #                 if not os.path.exists(output_dir + '/' + args.exp_name):
                #                     os.makedirs(output_dir + '/' + args.exp_name)
                #             save_path = output_dir + '/' + args.exp_name + '/_model.{}_{}'.format(self.state.global_step,
                #                                                                                   step)
                #             # logger.info('save to {}, epoch: {}'.format(save_path, self.state.epoch))
                #             logger.info('save to {}, epoch: {}'.format(save_path + '_adapter', self.state.epoch))
                #
                #             def safe_save_model_for_hf_trainer(trainer, output_dir):
                #                 """Collects the state dict and dump to disk."""
                #                 state_dict = trainer.model.state_dict()
                #
                #                 cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
                #                 del state_dict
                #                 trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
                #
                #             safe_save_model_for_hf_trainer(trainer=self, output_dir=save_path + '_adapter')
                if args.local_rank <= 0:
                        # unwrap_model(model).save_pretrained(save_path + '_adapter')
                        # self.tokenizer.save_pretrained(save_path + '_adapter')
                    if (step + 1) % print_every == 0:
                        self.his_info.lr = self.optimizer.param_groups[0]['lr']
                        self.his_info.update()
                        self.his_info.print_info(tokenizer=self.tokenizer, step=step)



                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                            version.parse(accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            # if args.no_zero:
            #     if args.local_rank <= 0:
            #         if not os.path.exists(output_dir + '/' + args.exp_name):
            #             os.makedirs(output_dir + '/' + args.exp_name)
            #         save_path = output_dir + '/' + args.exp_name + '/_model.{}_{}'.format(self.state.global_step, step)
            #         # logger.info('save to {}, epoch: {}'.format(save_path, self.state.epoch))
            #         logger.info('save to {}, epoch: {}'.format(save_path + '_epoch' + str(epoch), self.state.epoch))
            #         unwrap_model(model).save_pretrained(save_path + '_epoch' + str(epoch))
            #         self.tokenizer.save_pretrained(save_path + '_epoch' + str(epoch))
            if not args.not_save:
                if args.zero3:
                    output_dir = args.output_dir
                    if args.local_rank <= 0:
                        if not os.path.exists(output_dir + '/' + args.exp_name):
                            os.makedirs(output_dir + '/' + args.exp_name)
                    save_path = output_dir + '/' + args.exp_name + '/_model.{}_{}'.format(self.state.global_step, step)
                    # logger.info('save to {}, epoch: {}'.format(save_path, self.state.epoch))
                    logger.info('save to {}, epoch: {}'.format(save_path + '_epoch' + str(epoch), self.state.epoch))
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self.tokenizer.save_pretrained(save_path + '_epoch' + str(epoch))
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    state_dict = self.accelerator.get_state_dict(model)
                    unwrapped_model.save_pretrained(
                        save_path + '_epoch' + str(epoch), is_main_process=self.accelerator.is_main_process,
                        save_function=self.accelerator.save, state_dict=state_dict
                    )
                else:

                    output_dir = args.output_dir
                    if args.local_rank <= 0:
                        if not os.path.exists(output_dir + '/' + args.exp_name):
                            os.makedirs(output_dir + '/' + args.exp_name)
                    save_path = output_dir + '/' + args.exp_name + '/_model.{}_{}'.format(self.state.global_step,
                                step)

                    def safe_save_model_for_hf_trainer(trainer, output_dir):
                        """Collects the state dict and dump to disk."""
                        state_dict = trainer.model.state_dict()

                        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
                        del state_dict
                        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

                    safe_save_model_for_hf_trainer(trainer=self, output_dir=save_path + '_epoch' + str(epoch))


            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")


        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)





        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def compute_acc_and_loss(self, max_ids, labels_, mp, log_probs_all):
        '''
         Calculateing accuracy and loss while generating n-gram related positional information.
        '''
        tot_cont = 0
        dif_cont = 0
        # acc max_ids -> labels
        # loss sum of log_probs
        # log_probs_all # bs, seq_len vocab
        masked_probs = log_probs_all.gather(2, max_ids.unsqueeze(2)).squeeze()


        pred_acc = max_ids == labels_

        batch_p_numpy = mp.clone().detach().cpu().numpy()
        batch_2_gram_pos = []
        batch_3_gram_pos = []
        batch_4_gram_pos = []
        batch_n_gram_pos = []
        labels_np = labels_.cpu().clone().numpy()
        for k, p_n in enumerate(batch_p_numpy):
            cont_mask_num = 0
            _2_gram_pos = [0] * len(labels_np[k])
            _3_gram_pos = [0] * len(labels_np[k])
            _4_gram_pos = [0] * len(labels_np[k])
            _n_gram_pos = [0] * len(labels_np[k])
            for i in range(0, len(p_n)):
                if p_n[i] == 0:
                    break
                if i > 0 and p_n[i] == p_n[i - 1] + 1:
                    cont_mask_num += 1
                elif i == 0:  # 0 or not cont from last pos
                    cont_mask_num = 1
                else:
                    cont_mask_num = 1
                if p_n[i] + 1 < len(labels_np[k]) and labels_np[k][p_n[i] + 1] != -100:
                    if cont_mask_num >= 1:
                        _n_gram_pos[p_n[i] + 1] = 1
                    if cont_mask_num == 1:
                        _2_gram_pos[p_n[i] + 1] = 1
                    if cont_mask_num == 2:
                        _3_gram_pos[p_n[i] + 1] = 1
                    if cont_mask_num == 3:
                        _4_gram_pos[p_n[i] + 1] = 1


            batch_2_gram_pos.append(_2_gram_pos)
            batch_3_gram_pos.append(_3_gram_pos)
            batch_4_gram_pos.append(_4_gram_pos)
            batch_n_gram_pos.append(_n_gram_pos)
        batch_2_gram_pos = torch.tensor(batch_2_gram_pos).long().cuda()
        batch_3_gram_pos = torch.tensor(batch_3_gram_pos).long().cuda()
        batch_4_gram_pos = torch.tensor(batch_4_gram_pos).long().cuda()
        batch_n_gram_pos = torch.tensor(batch_n_gram_pos).long().cuda()

        # print(batch_2_gram_pos.shape)
        # print(log_probs_all.shape)
        probs = torch.exp(log_probs_all)
        entropy = -torch.sum(probs * log_probs_all, dim=-1) #bs, seq
        # assert 1==0
        uni_gram_pos = 1 - batch_n_gram_pos
        _1_gram_entropy = (entropy * uni_gram_pos).sum() / (uni_gram_pos.sum())
        _2_gram_entropy = (entropy * batch_2_gram_pos).sum() / (batch_2_gram_pos.sum())
        _3_gram_entropy = (entropy * batch_3_gram_pos).sum() / (batch_3_gram_pos.sum())
        _4_gram_entropy = (entropy * batch_4_gram_pos).sum() / (batch_4_gram_pos.sum())
        _2_gram_loss = -(masked_probs * batch_2_gram_pos).sum() / (batch_2_gram_pos.sum())
        _3_gram_loss = -(masked_probs * batch_3_gram_pos).sum() / (batch_3_gram_pos.sum())
        _4_gram_loss = -(masked_probs * batch_4_gram_pos).sum() / (batch_4_gram_pos.sum())
        _2_gram_acc = (pred_acc * batch_2_gram_pos).sum() / (batch_2_gram_pos.sum())
        _3_gram_acc = (pred_acc * batch_3_gram_pos).sum() / (batch_3_gram_pos.sum())
        _4_gram_acc = (pred_acc * batch_4_gram_pos).sum() / (batch_4_gram_pos.sum())
        if uni_gram_pos.sum() != 0:
            self.his_info.his_1_gram_ent.append(_1_gram_entropy.cpu())
        if batch_2_gram_pos.sum() != 0:
            self.his_info.his_2_gram_acc.append(_2_gram_acc.cpu())
            self.his_info.his_2_gram_loss.append(_2_gram_loss.cpu())
            self.his_info.his_2_gram_ent.append(_2_gram_entropy.cpu())
        if batch_3_gram_pos.sum() != 0:
            self.his_info.his_3_gram_acc.append(_3_gram_acc.cpu())
            self.his_info.his_3_gram_loss.append(_3_gram_loss.cpu())
            self.his_info.his_3_gram_ent.append(_3_gram_entropy.cpu())
        if batch_4_gram_pos.sum() != 0:
            self.his_info.his_4_gram_acc.append(_4_gram_acc.cpu())
            self.his_info.his_4_gram_loss.append(_4_gram_loss.cpu())
            self.his_info.his_4_gram_ent.append(_4_gram_entropy.cpu())
        return batch_2_gram_pos, batch_3_gram_pos, batch_4_gram_pos, batch_n_gram_pos

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.args.update_mask_rate:
            try:
                model.mask_policy.config.mask_rate = 0 + self.args.max_mask_rate * float(self.state.global_step) / ((self.max_steps * self.args.mask_rate_warmup_ratio) + 1)
                model.mask_policy.config.mask_rate = min(model.mask_policy.config.mask_rate, self.args.max_mask_rate)
                self.his_info.mask_rate = model.mask_policy.config.mask_rate
            except:
                model.module.mask_policy.config.mask_rate = 0 + self.args.max_mask_rate * float(self.state.global_step) / (
                            (self.max_steps * self.args.mask_rate_warmup_ratio) + 1)
                model.module.mask_policy.config.mask_rate = min(model.module.mask_policy.config.mask_rate, self.args.max_mask_rate)
                self.his_info.mask_rate = model.module.mask_policy.config.mask_rate

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        device = inputs['input_ids'].device
        ori_inputs = copy.deepcopy(inputs)

        model.train()
        batch_size = inputs['input_ids'].shape[0]
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.tokenizer = self.tokenizer
            model.tokenizer = self.tokenizer
        else:
            model.tokenizer = self.tokenizer
        gt_data = inputs['data_gt']
        ce_cands = gt_data
        def get_id_cands(cands):
            inputs_local = {}
            inputs_local['input_ids'],  inputs_local['labels'], inputs_local['attention_mask'], \
            inputs_local['label_mask'] = self.model.mask_policy.format_and_padding(inputs['data_src'], cands, device,
                                                                     self.args.tok_max_length, self.args.pad_front, format=self.args.instruct_format,
                                                                     tgt_max_len=self.args.tgt_max_length,
                                                                                tokenizer=self.tokenizer,
                                                                      instruct_type=self.args.instruct_type, cut_src=self.args.cut_src)
            return inputs_local
        start = time.time()
        inputs_ce = get_id_cands(ce_cands)

        model_return_dict = model(**inputs_ce)
        outputs, _, _, max_ids, _, _, labels_, _, log_probs_all, \
            _, _, masked_pos_non_shift, = model_return_dict[:12]

        self.compute_acc_and_loss(
            max_ids, labels_, masked_pos_non_shift, log_probs_all)

        forward_time = time.time()-start
        self.his_info.his_forward_time.append(forward_time)
        start = time.time()

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
