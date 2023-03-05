"""
Trainer class.
"""

import json
import os
import time
from collections import OrderedDict

import torch
import numpy as np

from tqdm import tqdm

from space.args import str2bool
from space.trainers.trainer import Trainer


class IntentTrainer(Trainer):

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Trainer")
        group.add_argument("--can_norm", type=str2bool, default=True,
                           help="Whether to use CAN normalization for probabilities predicted by model.")
        Trainer.add_cmdline_argument(parser, group=group)
        return group

    def __init__(self, model, to_tensor, hparams, reader=None):
        super(IntentTrainer, self).__init__(model, to_tensor, hparams, reader)
        self.example = hparams.example
        self.do_train = hparams.do_train
        self.do_infer = hparams.do_infer
        self.can_norm = hparams.can_norm

    def can_normalization(self, y_pred, y_true, ex_data_iter):
        # 预测结果，计算修正前准确率
        acc_original = np.mean([y_pred.argmax(1) == y_true])
        message = 'original acc: %s' % acc_original

        # 评价每个预测结果的不确定性
        k = 3
        y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
        y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
        y_pred_uncertainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)

        # 选择阈值，划分高、低置信度两部分
        # print(np.sort(y_pred_uncertainty)[-100:].tolist())
        threshold = 0.7
        y_pred_confident = y_pred[y_pred_uncertainty < threshold]
        y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]
        y_true_confident = y_true[y_pred_uncertainty < threshold]
        y_true_unconfident = y_true[y_pred_uncertainty >= threshold]

        # 显示两部分各自的准确率
        # 一般而言，高置信度集准确率会远高于低置信度的
        acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean() \
            if len(y_true_confident) else 0.
        acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean() \
            if len(y_true_unconfident) else 0.
        message += '   (%s) confident acc: %s' % (len(y_true_confident), acc_confident)
        message += '   (%s) unconfident acc: %s' % (len(y_true_unconfident), acc_unconfident)

        # 从训练集统计先验分布
        prior = np.zeros(self.func_model.num_intent)
        for _, (batch, batch_size) in ex_data_iter:
            for intent_label in batch['intent_label']:
                prior[intent_label] += 1.

        prior /= prior.sum()

        # 逐个修改低置信度样本，并重新评价准确率
        right, alpha, iters = 0, 1, 1
        for i, y in enumerate(y_pred_unconfident):
            Y = np.concatenate([y_pred_confident, y[None]], axis=0)
            for j in range(iters):
                Y = Y ** alpha
                Y /= Y.mean(axis=0, keepdims=True)
                Y *= prior[None]
                Y /= Y.sum(axis=1, keepdims=True)
            y = Y[-1]
            if y.argmax() == y_true_unconfident[i]:
                right += 1

        # 输出修正后的准确率
        acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)
        if len(y_pred_unconfident):
            message += '   new unconfident acc: %s' % (right / len(y_pred_unconfident))
        else:
            message += '   no unconfident predictions'
        message += '   final acc: %s' % acc_final
        return acc_original, acc_final, message

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
                if self.example and with_label:
                    current_dataset = train_label_iter.data_file_to_dataset[data_file]
                    example_batch = self.reader.retrieve_examples(dataset=current_dataset,
                                                                  labels=batch["intent_label"],
                                                                  inds=batch["ids"], task="intent")
                    example_batch = type(example_batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])),
                                                            example_batch.items()))
                    for k, v in example_batch.items():
                        batch[k] = v
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

        # compute accuracy for valid dataset
        accuracy = self.infer(data_iter=valid_label_iter, ex_data_iter=train_label_iter)

        # report summary message and save checkpoints
        self.save_and_log_message(report_for_unlabeled_data, cur_valid_metric=-accuracy)

    def infer(self, data_iter, num_batches=None, ex_data_iter=None):
        """
        Inference interface.
        """
        self.logger.info("Generation starts ...")
        infer_save_file = os.path.join(self.save_dir, f"infer_{self.epoch}.result.json")

        # Inference
        batch_cnt = 0
        pred, true = [], []
        outputs, labels = [], []
        begin_time = time.time()

        with torch.no_grad():
            if self.example:
                for _, (batch, batch_size) in tqdm(ex_data_iter, desc="Building train memory."):
                    batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                    result = self.model.infer(inputs=batch)
                    outputs.append(torch.from_numpy(result['features']))
                    labels += batch["intent_label"].tolist()

                mem = torch.cat(outputs, dim=0)
                mem = mem.cuda() if self.func_model.use_gpu else mem
                labels = torch.LongTensor(labels).unsqueeze(0)
                labels = labels.cuda() if self.func_model.use_gpu else labels
                self.logger.info(f"Memory size: {mem.size()}")

            for _, (batch, batch_size) in tqdm(data_iter, total=num_batches):
                batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                result = self.model.infer(inputs=batch)

                if self.example:
                    features = torch.from_numpy(result['features'])
                    features = features.cuda() if self.func_model.use_gpu else features
                    probs = torch.softmax(features.mm(mem.t()), dim=-1)
                    intent_probs = torch.zeros(probs.size(0), self.func_model.num_intent)
                    intent_probs = intent_probs.cuda() if self.func_model.use_gpu else intent_probs
                    intent_probs = intent_probs.scatter_add(-1, labels.repeat(probs.size(0), 1), probs)
                    intent_probs = intent_probs.cpu().detach().numpy()
                else:
                    intent_probs = result['intent_probs']

                if self.can_norm:
                    pred += [intent_probs]
                    true += batch["intent_label"].cpu().detach().tolist()
                else:
                    pred += np.argmax(intent_probs, axis=1).tolist()
                    true += batch["intent_label"].cpu().detach().tolist()

                batch_cnt += 1
                if batch_cnt == num_batches:
                    break

        if self.can_norm:
            true = np.array(true)
            pred = np.concatenate(pred, axis=0)
            acc_original, acc_final, message = self.can_normalization(y_pred=pred, y_true=true,
                                                                      ex_data_iter=ex_data_iter)
            accuracy = max(acc_original, acc_final)
            infer_results = {'accuracy': accuracy, 'pred_labels': pred.tolist(), 'message': message}
            metrics_message = f"Accuracy: {accuracy}   {message}"
        else:
            accuracy = sum(p == t for p, t in zip(pred, true)) / len(pred)
            infer_results = {'accuracy': accuracy, 'pred_labels': pred}
            metrics_message = f"Accuracy: {accuracy}"

        self.logger.info(f"Saved inference results to {infer_save_file}")
        with open(infer_save_file, "w") as fp:
            json.dump(infer_results, fp, indent=2)
        message_prefix = f"[Infer][{self.epoch}]"
        time_cost = f"TIME-{time.time() - begin_time:.3f}"
        message = "   ".join([message_prefix, metrics_message, time_cost])
        self.logger.info(message)
        return accuracy

    def track_and_log_message(self, metrics, batch_id, batch_size, num_batches, times, with_label):
        # track metrics
        batch_metrics_tracker = self.batch_metrics_tracker_label if with_label else self.batch_metrics_tracker_nolabel
        token_metrics_tracker = self.token_metrics_tracker_label if with_label else self.token_metrics_tracker_nolabel

        metrics = {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
                   for k, v in metrics.items()}
        mlm_num = metrics.pop("mlm_num", 0)

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

    def save_and_log_message(self, report_for_unlabeled_data, cur_valid_metric=None):
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
        assert cur_valid_metric is not None
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

            intent_loss, mlm, token_mlm, mlm_num, kl = metrics
            metrics = {}

            intent_loss = torch.mean(intent_loss)
            metrics['intent_loss'] = intent_loss
            loss = intent_loss

            if mlm is not None:
                mlm_num = torch.sum(mlm_num)
                token_mlm = torch.sum(mlm) * (batch_size / self.gpu) / mlm_num
                mlm = torch.mean(mlm)
                metrics['mlm_num'] = mlm_num
                metrics['token_mlm'] = token_mlm
                metrics['mlm'] = mlm
                loss = loss + (token_mlm if self.func_model.token_loss else mlm) * self.func_model.mlm_ratio

            if kl is not None:
                kl = torch.mean(kl)
                metrics['kl'] = kl
                loss = loss + kl * self.func_model.kl_ratio

            metrics['loss'] = loss

        assert 'loss' in metrics
        return metrics['loss'], metrics

    def load(self):
        """ load """
        def _load():
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

        if self.func_model.init_checkpoint is None:
            self.logger.info(f"Loaded no pre-train model !!!")
            return

        if self.do_train:
            _load()
            return

        if self.do_infer and not self.do_train:
            _load()
            train_file = f"{self.func_model.init_checkpoint}.train"
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
