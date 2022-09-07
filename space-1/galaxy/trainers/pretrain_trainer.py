"""
Pretrain Trainer class.
"""

import logging
import os
import sys
import time
from collections import OrderedDict

import torch

import numpy as np
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from galaxy.args import str2bool
from galaxy.data.data_loader import DataLoader
from galaxy.metrics.metrics_tracker import MetricsTracker


def get_logger(log_path, name="default"):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class PretrainTrainer(object):

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add the cmdline arguments of trainer. """
        group = parser.add_argument_group("Trainer")
        group.add_argument("--seed", type=int, default=11,
                           help="The number of seed to fix random operations.")
        group.add_argument("--gpu", type=int, default=0,
                           help="Whether to use gpu for running, default using cpu.")
        group.add_argument("--valid_metric_name", type=str, default="-loss",
                           help="The validation metric determining which checkpoint is the best.")
        group.add_argument("--num_epochs", type=int, default=10,
                           help="Total number of training epochs to perform.")
        group.add_argument("--save_dir", type=str, required=True,
                           help="The output directory where the model will be saved.")
        group.add_argument("--batch_size", type=int, default=8,
                           help="Total batch size for training/evaluation/inference.")
        group.add_argument("--log_steps", type=int, default=20,
                           help="The number of training steps to output current metrics "
                           "on past training dataset.")
        group.add_argument("--valid_steps", type=int, default=0,
                           help="The number of training steps to perform a evaluation "
                           "on validation datasets.")
        group.add_argument("--token_loss", type=str2bool, default=True,
                           help="Whether to update token loss or sentence loss.")
        group.add_argument("--save_checkpoint", type=str2bool, default=True,
                           help="Whether to save one checkpoints for each training epoch.")
        DataLoader.add_cmdline_argument(group)
        return group

    def __init__(self, model, to_tensor, hparams, reader=None, logger=None,
                 lr_scheduler=None, optimizer=None):
        self.model = model
        self.to_tensor = to_tensor

        self.is_decreased_valid_metric = hparams.valid_metric_name[0] == "-"
        self.valid_metric_name = hparams.valid_metric_name[1:]
        self.num_epochs = hparams.num_epochs
        self.save_dir = hparams.save_dir
        self.log_steps = hparams.log_steps
        self.valid_steps = hparams.valid_steps
        self.save_checkpoint = hparams.save_checkpoint
        self.weight_decay = hparams.weight_decay
        self.warmup_steps = hparams.warmup_steps
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        self.device = hparams.device
        self.world_size = hparams.world_size
        self.global_rank = hparams.global_rank
        self.local_rank = hparams.local_rank

        self.model = model
        self.func_model = self.model.module
        self.reader = reader
        self.tokenizer = reader.tokenizer

        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger = logger or get_logger(os.path.join(self.save_dir, "trainer.log"), "trainer")

        self.batch_metrics_tracker = MetricsTracker()
        self.token_metrics_tracker = MetricsTracker()

        self.best_valid_metric = float("inf" if self.is_decreased_valid_metric else "-inf")
        self.epoch = 0
        self.batch_num = 0

    def set_optimizers(self, num_training_steps_per_epoch):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        num_training_steps = num_training_steps_per_epoch * self.num_epochs
        num_warmup_steps = self.warmup_steps if self.warmup_steps >= 0 else int(num_training_steps * 0.1)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # reset optimizer and lr_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # log info
        self.logger.info("***** Running training *****")
        self.logger.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                         num_training_steps_per_epoch)
        self.logger.info("  Num Epochs = %d", self.num_epochs)
        self.logger.info("  Batch size  = %d", self.batch_size)
        self.logger.info("  Total optimization steps = %d", num_training_steps)
        self.logger.info("  Total warmup steps = %d", num_warmup_steps)
        self.logger.info(f"Distributed info: world_size and global rank: " +
                         str(self.world_size) + "  " + str(self.global_rank))

    def train_epoch(self, train_iter):
        """
        Train an epoch.
        """
        self.epoch += 1
        self.batch_metrics_tracker.clear()
        self.token_metrics_tracker.clear()
        num_batches = len(train_iter)
        times = []
        for batch_id, (batch, batch_size) in enumerate(train_iter, 1):
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
            batch["epoch"] = self.epoch
            batch["num_steps"] = self.batch_num

            # Do a training iteration
            start_time = time.time()
            metrics = self.model(batch, is_training=True)

            loss = metrics["loss"]
            self.func_model._optimize(loss, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)

            metrics = {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
                       for k, v in metrics.items()}
            token_num = metrics.pop("token_num", None)
            bce_num = metrics.pop("bce_num", None)
            elapsed = time.time() - start_time
            times.append(elapsed)

            batch_metrics = {k: v for k, v in metrics.items() if "token" not in k}
            token_metrics = {k: v for k, v in metrics.items() if "token" in k}
            self.batch_metrics_tracker.update(batch_metrics, batch_size)
            self.token_metrics_tracker.update(token_metrics, token_num)
            self.batch_num += 1

            if self.log_steps > 0 and batch_id % self.log_steps == 0:
                batch_metrics_message = self.batch_metrics_tracker.value()
                token_metrics_message = self.token_metrics_tracker.value()
                message_prefix = f"[Train][{self.epoch}][{batch_id}/{num_batches}]"
                avg_time = f"AVG_Time-{sum(times[-self.log_steps:]) / self.log_steps:.3f}"
                message = "   ".join([message_prefix, batch_metrics_message, token_metrics_message,
                                      avg_time])
                self.logger.info(message)

        batch_metrics_message = self.batch_metrics_tracker.summary()
        token_metrics_message = self.token_metrics_tracker.summary()
        message_prefix = f"[Valid][{self.epoch}]"
        message = "   ".join([message_prefix, batch_metrics_message, token_metrics_message])
        self.logger.info(message)

        cur_valid_metric = self.batch_metrics_tracker.get(self.valid_metric_name)
        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric
        if is_best:
            self.best_valid_metric = cur_valid_metric
        if self.local_rank == -1 or self.global_rank == 0:
            self.save(is_best)
        return

    def save(self, is_best=False):
        """ save """
        train_state = {"epoch": self.epoch,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()

        # Save checkpoint
        if self.save_checkpoint:
            model_file = os.path.join(self.save_dir, f"state_epoch_{self.epoch}.model")
            torch.save(self.model.state_dict(), model_file)
            self.logger.info(f"Saved model state to '{model_file}'")

            train_file = os.path.join(self.save_dir, f"state_epoch_{self.epoch}.train")
            torch.save(train_state, train_file)
            self.logger.info(f"Saved train state to '{train_file}'")

        # Save current best model
        if is_best:
            best_model_file = os.path.join(self.save_dir, "best.model")
            torch.save(self.model.state_dict(), best_model_file)
            best_train_file = os.path.join(self.save_dir, "best.train")
            torch.save(train_state, best_train_file)
            self.logger.info(
                f"Saved best model state to '{best_model_file}' with new best valid metric "
                f"{self.valid_metric_name.upper()}={self.best_valid_metric:.3f}")

    def load(self):
        """ load """
        def _load_model_state():
            model_state_dict = torch.load(f'{self.func_model.init_checkpoint}.model',
                                          map_location=lambda storage, loc: storage)

            if 'module.' in list(model_state_dict.keys())[0]:
                new_model_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    assert k[:7] == 'module.'
                    new_model_state_dict[k[7:]] = v
                model_state_dict = new_model_state_dict

            new_model_state_dict = OrderedDict()
            parameters = {name: param for name, param in self.func_model.named_parameters()}
            for name, param in model_state_dict.items():
                if name in parameters:
                    if param.shape != parameters[name].shape:
                        assert hasattr(param, "numpy")
                        arr = param.numpy()
                        z = np.random.normal(scale=self.func_model.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        if name == 'embedder.token_embedding.weight':
                            z[-param.shape[0]:] = arr
                            print(f"part of parameter({name}) random normlize initialize")
                        else:
                            if z.shape[0] < param.shape[0]:
                                z = arr[:z.shape[0]]
                                print(f"part of parameter({name}) are dropped")
                            else:
                                z[:param.shape[0]] = arr
                                print(f"part of parameter({name}) random normlize initialize")
                        dtype, device = param.dtype, param.device
                        z = torch.tensor(z, dtype=dtype, device=device)
                        new_model_state_dict[name] = z
                    else:
                        new_model_state_dict[name] = param
                else:
                    print(f"parameter({name}) are dropped")
            model_state_dict = new_model_state_dict

            for name in parameters:
                if name not in model_state_dict:
                    if parameters[name].requires_grad:
                        print(f"parameter({name}) random normlize initialize")
                        z = np.random.normal(scale=self.func_model.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        dtype, device = parameters[name].dtype, parameters[name].device
                        model_state_dict[name] = torch.tensor(z, dtype=dtype, device=device)
                    else:
                        model_state_dict[name] = parameters[name]

            self.func_model.load_state_dict(model_state_dict)
            self.logger.info(f"Loaded model state from '{self.func_model.init_checkpoint}.model'")

        def _load_train_state():
            train_file = f"{self.func_model.init_checkpoint}.train"
            if os.path.exists(train_file):
                train_state_dict = torch.load(train_file, map_location=lambda storage, loc: storage)
                self.epoch = train_state_dict["epoch"]
                self.best_valid_metric = train_state_dict["best_valid_metric"]
                if self.optimizer is not None and "optimizer" in train_state_dict:
                    self.optimizer.load_state_dict(train_state_dict["optimizer"])
                if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
                    self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
                self.logger.info(
                    f"Loaded train state from '{train_file}' with (epoch-{self.epoch} "
                    f"best_valid_metric={self.best_valid_metric:.3f})")
            else:
                self.logger.info(f"Loaded no train state")

        if self.func_model.init_checkpoint is None:
            self.logger.info(f"Loaded no model !!!")
            return

        _load_model_state()
        _load_train_state()
