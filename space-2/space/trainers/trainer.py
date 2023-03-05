"""
Trainer class.
"""

import json
import logging
import os
import sys
import time
from collections import OrderedDict

import torch

import numpy as np
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from space.args import str2bool
from space.data.data_loader import DataLoader
from space.metrics.metrics_tracker import MetricsTracker
from space.metrics.metrics import bleu
from space.metrics.metrics import distinct
from space.modules.subspace import Subspace


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


def evaluate_generation_result(results):
    tgt = [result["tgt"].split(" ") for result in results]
    pred = [result["preds"][np.argmax(result["scores"])]
            if isinstance(result["preds"], list)
            else result["preds"]
            for result in results]
    pred = [p.split(" ") for p in pred]
    metrics = {}
    metrics_tracker = MetricsTracker()

    bleu1, bleu2 = bleu(pred, tgt)
    metrics.update({"bleu_1": bleu1, "bleu_2": bleu2})

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(pred)
    metrics.update({"intra_dist_1": intra_dist1,
                    "intra_dist_2": intra_dist2,
                    "inter_dist_1": inter_dist1,
                    "inter_dist_2": inter_dist2})

    avg_len = sum(map(len, pred)) / len(pred)
    metrics.update({"len": avg_len})

    metrics_tracker.update(metrics, num_samples=1)  # 一次更新所有数据的指标到位，没有累积更新，故num_sample取为1
    return metrics_tracker


class Trainer(object):

    @classmethod
    def add_cmdline_argument(cls, parser, group=None):
        """ Add the cmdline arguments of trainer. """
        group = parser.add_argument_group("Trainer") if group is None else group
        group.add_argument("--seed", type=int, default=11,
                           help="The number of seed to fix random operations.")
        group.add_argument("--gpu", type=int, default=0,
                           help="Whether to use gpu for running, default using cpu.")
        group.add_argument("--use_data_distributed", type=str2bool, default=False,
                           help="Whether to use data distributed for parallel training.")
        group.add_argument("--valid_metric_name", type=str, default="-loss",
                           help="The validation metric determining which checkpoint is the best.")
        group.add_argument("--num_epochs", type=int, default=10,
                           help="Total number of training epochs to perform.")
        group.add_argument("--save_dir", type=str, required=True,
                           help="The output directory where the model will be saved.")
        group.add_argument("--batch_size_label", type=int, default=8,
                           help="Total batch size for training/evaluation/inference of labeled data.")
        group.add_argument("--batch_size_nolabel", type=int, default=8,
                           help="Total batch size for training/evaluation/inference of unlabeled data.")
        group.add_argument("--log_steps", type=int, default=20,
                           help="The number of training steps to output current metrics "
                           "on past training dataset.")
        group.add_argument("--valid_steps", type=int, default=0,
                           help="The number of training steps to perform a evaluation "
                           "on validation datasets.")
        group.add_argument("--token_loss", type=str2bool, default=True,
                           help="Whether to update token loss or sentence loss.")
        group.add_argument("--warmup_steps", type=int, default=-1,
                           help="The number of warmup steps for lr.")
        group.add_argument("--save_checkpoint", type=str2bool, default=True,
                           help="Whether to save one checkpoints for each training epoch.")
        group.add_argument("--save_summary", type=str2bool, default=False,
                           help="Whether to save metrics summary for visualDL module.")
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
        self.save_summary = hparams.save_summary
        self.learning_method = hparams.learning_method
        self.weight_decay = hparams.weight_decay
        self.warmup_steps = hparams.warmup_steps
        self.batch_size_label = hparams.batch_size_label
        self.batch_size_nolabel = hparams.batch_size_nolabel
        self.gpu = hparams.gpu
        self.lr = hparams.lr

        self.model = model
        self.func_model = self.model.module if self.gpu > 1 else self.model
        self.reader = reader
        self.tokenizer = reader.tokenizer

        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger = logger or get_logger(os.path.join(self.save_dir, "trainer.log"), "trainer")

        self.batch_metrics_tracker_label = MetricsTracker()
        self.token_metrics_tracker_label = MetricsTracker()
        self.batch_metrics_tracker_nolabel = MetricsTracker()
        self.token_metrics_tracker_nolabel = MetricsTracker()

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
        self.logger.info(f"***** Running training: {self.learning_method} *****")
        self.logger.info(f"  Subspaces = {Subspace.subspaces}")
        self.logger.info(f"  Triggered Subspaces = {self.func_model.subspace.trigger_subspaces if self.func_model.with_project else []}")
        self.logger.info("  Num Epochs = %d", self.num_epochs)
        self.logger.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d", num_training_steps_per_epoch)
        self.logger.info("  Batch size for labeled data = %d", self.batch_size_label)
        self.logger.info("  Batch size for unlabeled data = %d", self.batch_size_nolabel)
        self.logger.info("  Total optimization steps = %d", num_training_steps)
        self.logger.info("  Total warmup steps = %d", num_warmup_steps)
        self.logger.info(f"************************************")

    def train(self, train_label_iter, train_nolabel_iter=None, valid_label_iter=None, valid_nolabel_iter=None):
        # begin training
        num_epochs = self.num_epochs - self.epoch
        for epoch in range(num_epochs):
            self.train_epoch(train_label_iter=train_label_iter, train_nolabel_iter=train_nolabel_iter,
                             valid_label_iter=valid_label_iter, valid_nolabel_iter=valid_nolabel_iter)

    def train_epoch(self, train_label_iter, train_nolabel_iter, valid_label_iter, valid_nolabel_iter):
        """
        Train an epoch.
        """
        times = []
        self.epoch += 1
        self.batch_metrics_tracker_label.clear()
        self.token_metrics_tracker_label.clear()
        self.batch_metrics_tracker_nolabel.clear()
        self.token_metrics_tracker_nolabel.clear()

        num_label_batches = len(train_label_iter)
        num_nolabel_batches = len(train_nolabel_iter) if train_nolabel_iter is not None else 0
        num_batches = max(num_label_batches, num_nolabel_batches)

        train_label_iter_loop = iter(train_label_iter)
        train_nolabel_iter_loop = iter(train_nolabel_iter) if train_nolabel_iter is not None else None
        report_for_unlabeled_data = True if train_nolabel_iter is not None else False

        for batch_id in range(1, num_batches + 1):
            # Do a training iteration
            start_time = time.time()
            batch_list, batch_size_list, with_label_list, loss_list, metrics_list = [], [], [], [], []
            data_file_list = []

            # collect batch for labeled data
            try:
                data_file_label, (batch_label, batch_size_label) = next(train_label_iter_loop)
            except StopIteration:
                train_label_iter_loop = iter(train_label_iter)
                data_file_label, (batch_label, batch_size_label) = next(train_label_iter_loop)
            batch_list.append(batch_label)
            batch_size_list.append(batch_size_label)
            with_label_list.append(True)
            data_file_list.append(data_file_label)

            # collect batch for unlabeled data
            if train_nolabel_iter is not None:
                try:
                    data_file_nolabel, (batch_nolabel, batch_size_nolabel) = next(train_nolabel_iter_loop)
                except StopIteration:
                    train_nolabel_iter_loop = iter(train_nolabel_iter)
                    data_file_nolabel, (batch_nolabel, batch_size_nolabel) = next(train_nolabel_iter_loop)
                batch_list.append(batch_nolabel)
                batch_size_list.append(batch_size_nolabel)
                with_label_list.append(False)
                data_file_list.append(data_file_nolabel)

            # forward labeled batch and unlabeled batch and collect outputs, respectively
            for (batch, batch_size, with_label, data_file) in \
                    zip(batch_list, batch_size_list, with_label_list, data_file_list):
                batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                batch["epoch"] = self.epoch
                batch["num_steps"] = self.batch_num
                metrics = self.model(batch, is_training=True, with_label=with_label, data_file=data_file)
                loss, metrics = self.balance_metrics(metrics=metrics, batch_size=batch_size)
                loss_list.append(loss)
                metrics_list.append(metrics)

            # combine loss for labeled data and unlabeled data
            # TODO change the computation of combined loss of labeled batch and unlabeled batch
            loss = loss_list[0] if len(loss_list) == 1 else loss_list[0] + loss_list[1]

            # optimization procedure
            self.func_model._optimize(loss, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
            elapsed = time.time() - start_time
            times.append(elapsed)
            self.batch_num += 1

            # track metrics and log temporary message
            for (batch_size, metrics, with_label) in zip(batch_size_list, metrics_list, with_label_list):
                self.track_and_log_message(metrics=metrics, batch_id=batch_id, batch_size=batch_size,
                                           num_batches=num_batches, times=times, with_label=with_label)

            # evaluate
            if self.valid_steps > 0 and valid_label_iter is not None and valid_nolabel_iter is not None \
                    and batch_id % self.valid_steps == 0:
                self.evaluate(data_label_iter=valid_label_iter, data_nolabel_iter=valid_nolabel_iter)

        # report summary message and save checkpoints
        self.save_and_log_message(report_for_unlabeled_data=report_for_unlabeled_data)

    def evaluate(self, data_label_iter, data_nolabel_iter, need_save=True):
        """
        Evaluation interface
        和训练时一样，产生整个src+tgt端对应的输出，且tgt端生成也是完全teacher forcing

        @param : data_iter
        @type : DataLoader

        @param : need_save
        @type : bool
        """
        # Evaluation
        begin_time = time.time()
        batch_metrics_tracker = MetricsTracker()
        batch_metrics_label_tracker = MetricsTracker()
        batch_metrics_nolabel_tracker = MetricsTracker()
        with torch.no_grad():
            for batch, batch_size in data_label_iter:
                batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                pass
            for batch, batch_size in data_nolabel_iter:
                batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                pass

        batch_metrics_message = batch_metrics_tracker.summary()
        batch_metrics_label_message = batch_metrics_label_tracker.summary()
        batch_metrics_nolabel_message = batch_metrics_nolabel_tracker.summary()
        message_prefix = f"[Test]"
        time_cost = f"TIME-{time.time() - begin_time:.3f}"
        message = "   ".join([message_prefix, batch_metrics_message, batch_metrics_label_message,
                              batch_metrics_nolabel_message, time_cost])
        self.logger.info(message)

        if need_save:
            pass

        return

    def infer(self, data_iter, num_batches=None):
        """
        Inference interface.
        """
        self.logger.info("Generation starts ...")
        infer_save_file = os.path.join(self.save_dir, f"infer_{self.epoch}.case.json")

        # Inference
        batch_cnt = 0
        features, ids = [], []
        begin_time = time.time()

        with torch.no_grad():
            for _, (batch, batch_size) in tqdm(data_iter, total=num_batches):
                batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                result = self.model.infer(inputs=batch)

                features += result['features'].tolist()
                ids += result['ids'].tolist()

                batch_cnt += 1
                if batch_cnt == num_batches:
                    break

        infer_results = {'ids': ids, 'features': features}
        self.logger.info(f"Saved inference results to {infer_save_file}")
        with open(infer_save_file, "w") as fp:
            json.dump(infer_results, fp, indent=2)
        message_prefix = f"[Infer][{self.epoch}]"
        time_cost = f"TIME-{time.time() - begin_time:.3f}"
        message = "   ".join([message_prefix, time_cost])
        self.logger.info(message)
        return

    def track_and_log_message(self, metrics, batch_id, batch_size, num_batches, times, with_label):
        # track metrics
        batch_metrics_tracker = self.batch_metrics_tracker_label if with_label else self.batch_metrics_tracker_nolabel
        token_metrics_tracker = self.token_metrics_tracker_label if with_label else self.token_metrics_tracker_nolabel

        metrics = {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
                   for k, v in metrics.items()}
        mlm_num = metrics.pop("mlm_num", None)

        batch_metrics = {k: v for k, v in metrics.items() if "token" not in k}
        token_metrics = {k: v for k, v in metrics.items() if "token" in k}
        batch_metrics_tracker.update(batch_metrics, batch_size)
        token_metrics_tracker.update(token_metrics, mlm_num)

        # log message
        if self.log_steps > 0 and batch_id % self.log_steps == 0:
            batch_metrics_message = batch_metrics_tracker.value()
            token_metrics_message = token_metrics_tracker.value()
            label_prefix = 'Labeled' if with_label else 'Unlabeled'
            message_prefix = f"[Train][{self.epoch}][{batch_id}/{num_batches}][{label_prefix}]"
            avg_time = f"AVG_Time-{sum(times[-self.log_steps:]) / self.log_steps:.3f}"
            message = "   ".join([message_prefix, batch_metrics_message, token_metrics_message,
                                  avg_time])
            self.logger.info(message)

    def save_and_log_message(self, report_for_unlabeled_data):
        # report message
        batch_metrics_message = self.batch_metrics_tracker_label.summary()
        token_metrics_message = self.token_metrics_tracker_label.summary()
        message_prefix = f"[Valid][{self.epoch}][Labeled]"
        message = "   ".join([message_prefix, batch_metrics_message, token_metrics_message])
        self.logger.info(message)
        if report_for_unlabeled_data:
            batch_metrics_message = self.batch_metrics_tracker_nolabel.summary()
            token_metrics_message = self.token_metrics_tracker_nolabel.summary()
            message_prefix = f"[Valid][{self.epoch}][Unlabeled]"
            message = "   ".join([message_prefix, batch_metrics_message, token_metrics_message])
            self.logger.info(message)

        # save checkpoints
        if report_for_unlabeled_data:
            # TODO change the computation of "cur_valid_metric"
            cur_valid_metric = self.batch_metrics_tracker_label.get(self.valid_metric_name) + \
                               self.batch_metrics_tracker_nolabel.get(self.valid_metric_name)
        else:
            cur_valid_metric = self.batch_metrics_tracker_label.get(self.valid_metric_name)
        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric
        if is_best:
            self.best_valid_metric = cur_valid_metric
        self.save(is_best)

    def balance_metrics(self, metrics, batch_size):
        if self.gpu > 1:
            for metric in metrics:
                if metric is not None:
                    assert len(metric) == self.gpu
            con, mlm, token_mlm, mlm_num = metrics
            metrics = {}
            loss = 0.

            if mlm is not None:
                mlm_num = torch.sum(mlm_num)
                token_mlm = torch.sum(mlm) * (batch_size / self.gpu) / mlm_num
                mlm = torch.mean(mlm)
                metrics['mlm_num'] = mlm_num
                metrics['token_mlm'] = token_mlm
                metrics['mlm'] = mlm
                loss = loss + (token_mlm if self.func_model.token_loss else mlm) * self.func_model.mlm_ratio

            if con is not None:
                con = torch.mean(con)
                metrics['con'] = con
                loss = loss + con

            metrics['loss'] = loss

        assert 'loss' in metrics
        return metrics['loss'], metrics

    def save(self, is_best=False):
        """ save """
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
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
        if self.func_model.init_checkpoint is None:
            # train from scratch
            # init_checkpoint: None
            self.logger.info(f"Loaded no pre-train model !!!")
            return

        if ('/best' not in self.func_model.init_checkpoint) and \
                ('/state_epoch' not in self.func_model.init_checkpoint):
            # load pre-trained model, then begin training
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
            self.logger.info(f"Loaded pre-train model from '{self.func_model.init_checkpoint}'")
            return

        if os.path.isfile(f'{self.func_model.init_checkpoint}.model'):
            # load fine-tuned model, then evaluate / train (continue fine-tune)
            # init_checkpoint: 'outputs/${dataset}/' + ('best.model' / 'state_epoch_*.model')
            file_prefix = self.func_model.init_checkpoint
            model_file = f"{file_prefix}.model"
            train_file = f"{file_prefix}.train"

            model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            if 'module.' in list(model_state_dict.keys())[0] and self.gpu <= 1:
                new_model_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    assert k[:7] == 'module.'
                    new_model_state_dict[k[7:]] = v
                model_state_dict = new_model_state_dict
            try:
                self.model.load_state_dict(model_state_dict)
            except:
                self.func_model.load_state_dict(model_state_dict)
                self.logger.info(f"Warn: Using {self.gpu} gpus, but try to load non-DataParallel model"
                                 f" into DataParallel model!")
            self.logger.info(f"Loaded model state from '{model_file}'")

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
