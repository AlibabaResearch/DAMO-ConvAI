import os
import time
import math
import torch
from torch.utils.data import DistributedSampler
import torch.distributed as torch_dist
from torch.nn.parallel import DistributedDataParallel

from evaluate.evaluator import Evaluator
from uda import utils
from tools.logger import init_logger
from uda.models import load_model
from optimizer import init_optimizer
from lr_scheduler import init_lr_scheduler
from data import init_dataset
from data.dataloader import init_dataloader
from data.noise_processor import NoiseProcessor


class Trainer(object):
    def __init__(self, model, model_config, tokenizer, special_tokens, optimizer, train_dataloader, train_sampler,
                 logger, args, rank=0, word_size=1):
        super(Trainer, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.optimizer = optimizer
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.dist_train = args.dist_train
        self.distributed_env = args.distributed_env

        self.logger = logger
        self.args = args

        #
        self.matrix = args.matrix
        self.model_name = args.model_name
        self.max_epochs = args.max_epochs
        self.enable_epoch_training = self.max_epochs > 0
        self.early_stopping_patience = args.early_stopping_patience
        self.enable_iterable_dataset = args.dataset_style == 'iterable'

        self.start_eval_from = args.start_eval_from
        self.eval_every = args.eval_every

        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.eval_max_target_length = args.eval_max_target_length
        self.val_metric = args.val_metric
        self.max_keep_checkpoints = args.max_keep_checkpoints
        self.report_every = args.report_every
        self.warmup_steps = args.warmup_steps
        self.max_grad_norm = args.max_grad_norm

        self.learning_rate = args.learning_rate

        self.lr_scheduler = init_lr_scheduler(lr_scheduler_type=args.lr_scheduler, optimizer=optimizer,
                                              warmup_steps=self.warmup_steps, max_steps=self.max_steps)

        self.update_freq = args.update_freq
        assert self.update_freq > 0
        self.saved_dir = args.saved_dir
        self.experiment_dir = os.path.join(self.saved_dir, args.experiment_name)

        self.saved_tokenizer_dir = os.path.join(self.experiment_dir, 'config_and_tokenizer')
        self.model_saved_dir = os.path.join(self.experiment_dir, 'models')
        self.generated_text_dir = os.path.join(self.experiment_dir, 'generated_texts')

        if rank == 0:
            for path in [self.saved_dir, self.experiment_dir, self.saved_tokenizer_dir, self.model_saved_dir,
                         self.generated_text_dir]:
                if not os.path.exists(path):
                    os.mkdir(path)

            model_config.save_pretrained(self.saved_tokenizer_dir)
            self.tokenizer.save_pretrained(self.saved_tokenizer_dir)
            # self.metric_file_fin = open(os.path.join(self.experiment_dir, 'metric.json'), 'w')

            # if rank == 0:
            self.evaluator = Evaluator.init_evaluator(args=args, model_name=self.model_name, tokenizer=self.tokenizer,
                                                      special_tokens=special_tokens,
                                                      generated_text_dir=self.generated_text_dir)
        else:
            self.evaluator = None
            # self.metric_file_fin = None

        logger.info("Rank: {}".format(rank))
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(type=args.device, index=rank) if self.dist_train else torch.device(args.device)
        self.rank = rank
        self.word_size = word_size

        # common values
        self.start_time = time.time()
        self.trained_steps = 0
        self.trained_epochs = 0
        self.best_epoch = 0  # not work when
        self.best_step = 0
        self.best_val_metric = 0.0
        self.best_obleu = 0.0
        self.keep_checkpoints = []  # [(checkpoint_file_name, eval_metric_score)] # desc order by eval_metric score
        self.evaluate_metric_results = dict()
        self.steps_one_epoch = self.obtrain_steps_for_one_epoch()

    @classmethod
    def init_trainer(self, args):
        logger = init_logger(__name__)

        if args.dist_train:
            self.distributed_data_parallel_setup(args.local_rank)
            rank = torch_dist.get_rank()
            word_size = torch_dist.get_world_size()
            # print("@@@@@@", torch.cuda.device_count())
            logger.info("Enable Distributed data paralle training, word size: {}".format(word_size))
        else:
            rank = 0
            word_size = 1

        tokenizer, config, model = load_model(tokenizer_path=args.tokenizer_path,
                                              model_name=args.model_name, model_path=args.init_model_path,
                                              args=args)
        model.to(torch.device(args.device, rank)) if args.dist_train else model.to(args.device)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        special_tokens = []
        if args.special_token_path is not None and os.path.exists(args.special_token_path):
            special_tokens = utils.read_text_file(args.special_token_path)
            n_added_token = tokenizer.add_tokens(special_tokens)
            if n_added_token > 0:
                logger.info("Add {} tokens".format(n_added_token))
                for special_token in special_tokens:
                    logger.info("- {}".format(special_token))
                oral_vocab = config.vocab_size
                model.resize_token_embeddings(oral_vocab + n_added_token)
                logger.info(
                    "Resize the token embeddings of model, from {} to {}".format(oral_vocab,
                                                                                 oral_vocab + n_added_token))

        # if rank == 0:
            # logger.info(model)
        optimizer = init_optimizer(model=model, learner=args.learner.lower(), learning_rate=float(args.learning_rate))
        noise_processor = None
        if args.train_type.lower() == 'pretrain':
            noise_processor = NoiseProcessor.init_noise_processor(
                extra_tokens=tokenizer.additional_special_tokens,
                args=args,
                random_delete_rate=args.random_delete_rate,
                noise_types=args.noise_types,
                noise_type_rates=args.noise_type_rates,
                noise_task_source_prefix=args.noise_task_source_prefix,
                random_allocation_mask=args.random_allocation_mask)

        train_dataset = init_dataset(
            data_dir=args.train_file_src,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            datatype=args.datatype,
            max_inp_len=args.max_source_length,
            max_target_len=args.max_target_length,
            n_example=args.n_train_example,
            enable_uda_relative_pos=args.enable_uda_relative_pos,
            data_processor=args.data_processor,
            task_source_prefix=args.task_source_prefix,
            noise_processor=noise_processor,
            rank=rank, num_gpus=word_size, dataset_style=args.dataset_style,
            position_style = args.position_style
        )

        train_sampler = None
        if args.dist_train:
            if getattr(args, 'distributed_env', 'distributed_data_parallel') == 'distributed_data_parallel':
                if args.dataset_style == 'map':
                    train_sampler = DistributedSampler(train_dataset)

        train_dataloader = init_dataloader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            num_workers=args.train_num_workers,
            dist_train=args.dist_train,
            pin_memory=args.train_pin_memory,
            shuffle=train_sampler is None and args.dataset_style == 'map', sampler=train_sampler
        )
        # for data in train_dataloader:
        #     data

        if args.dist_train:
            # model = DistributedDataParallel(module=model,
            #                                             device_ids=[args.local_rank],
            #                                             output_device=args.local_rank,
            #                                             find_unused_parameters=True)
            model = DistributedDataParallel(module=model,
                                            device_ids=[rank],
                                            output_device=rank,
                                            find_unused_parameters=False)

        trainer = self(model=model, model_config=config, tokenizer=tokenizer, special_tokens=special_tokens,
                       optimizer=optimizer,
                       train_dataloader=train_dataloader,
                       train_sampler=train_sampler,
                       logger=logger, args=args, rank=rank, word_size=word_size)
        return trainer

    """
    @staticmethod
    def init_optimizer(model, args):

        learner = args.learner.lower()
        learning_rate = float(args.learning_rate)
        if learner == 'adamw':
            optimizer = AdamW(params=model.parameters(), lr=learning_rate)
        elif learner == 'adam':
            optimizer = torch_optim.Adam(params=model.parameters(), lr=learning_rate)
        elif learner == 'adagrad':
            optimizer = torch_optim.Adagrad(params=model.parameters(), lr=learning_rate)
        elif learner == 'adafactor':
            optimizer = Adafactor(params=model.parameters(), lr=learning_rate,
                                  scale_parameter=False, relative_step=False)
        elif learner == 'fairseq_adafactor':
            optimizer = Fairseq_Adafactor(params=model.parameters(), lr=learning_rate,
                                          scale_parameter=False, relative_step=False)
        else:
            raise NotImplemented

        return optimizer
    
    def get_lr_scheduler(self, lr_scheduler_type):

        if lr_scheduler_type == 'none':
            return None

        elif lr_scheduler_type == 'linear':
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            raise NotImplemented

        return lr_scheduler
    """

    @staticmethod
    def distributed_data_parallel_setup(local_rank, global_rank=None, world_size=None):
        # torch_dist.init_process_group(backend="nccl", world_size=world_size, rank=global_rank)
        torch_dist.init_process_group(backend="nccl")
        # torch_dist.init_process_group(backend="gloo")
        # torch_dist.barrier()
        torch.cuda.set_device(local_rank)

    @staticmethod
    def distributed_data_parallel_cleanup():
        torch_dist.destroy_process_group()

    def training_stop_check(self):
        """

        :return: if True: stopping
        """
        if self.enable_epoch_training:
            return self.trained_epochs > self.max_epochs
        else:
            return self.trained_steps > self.max_steps

    def evaluate_check(self):
        """

        :return: if True: executing evaluating
        """
        if self.rank != 0:
            return False

        if self.enable_epoch_training:
            if self.trained_epochs < self.max_epochs:
                return self.trained_epochs >= self.start_eval_from and self.trained_epochs % self.eval_every == 0
            elif self.trained_epochs >= self.max_epochs:
                # evaluate the last epoch
                return True
            else:
                return False
        else:
            if self.trained_steps < self.max_steps:
                return self.trained_steps >= self.start_eval_from and self.trained_steps % self.eval_every == 0
            else:
                # evaluate the last epoch
                return True

    def train(self):
        # is_stopping = self.evaluate()
        # for epoch in range(self.max_epochs):
        while not self.training_stop_check():
            # training
            # losses = 0.5
            if self.dist_train:
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(
                        self.trained_epochs)  # make shuffling work properly across multiple epochs.

            # steps, losses = self.train_one_epoch()
            is_stopping = self.train_one_epoch()
            if self.enable_epoch_training:
                is_stopping = self.evaluate()
                if is_stopping:
                    break
            else:  # epoch training
                if is_stopping:
                    break

        if self.rank == 0:
            utils.write_to_json_file_by_line(self.evaluate_metric_results,
                                             os.path.join(self.experiment_dir, 'metric.json'))

        self.logger.info(
            "Finished training, the best val metric is {}, at {} epoch, {} steps".format(self.best_val_metric,
                                                                                         self.best_epoch,
                                                                                         self.best_step))

    @staticmethod
    def sort(list_pairs, reverse=True):
        return sorted(list_pairs, key=lambda item: item[1], reverse=reverse)

    def step(self, batch):
        # example_ids = batch['id']
        # target_text = batch['target_text']
        # target_text_lens = [len(text.split()) for text in target_text]
        input_ids = batch['enc_inp'].to(self.device)
        enc_attention_mask = batch['enc_attention_mask'].to(self.device)

        decoder_input_ids = batch['dec_inp'].to(self.device)
        labels = batch['label'].to(self.device)
        # self.logger.info("input_ids: {}, decoder_input_ids: {}, ids: {}, target_text_lens: {},"
        #                  "{}".format(input_ids.size(), decoder_input_ids.size(), example_ids, target_text_lens,
        #                              self.device))
        # decoder_input_ids = self.model._shift_right(labels)

        # assert False
        if self.model_name == 'uda':
            struct_attention = batch['struct_attention'].to(self.device)
            linear_relative_position_matrix = batch['linear_relative_position_matrix']
            if linear_relative_position_matrix is not None:
                linear_relative_position_matrix = linear_relative_position_matrix.to(self.device)

            if self.matrix == "nop":
                model_outputs = self.model(input_ids=input_ids,
                                       attention_mask=enc_attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       labels=labels,
                                       struct_attention=struct_attention,
                                           )
            elif self.matrix == "noa":
                model_outputs = self.model(input_ids=input_ids,
                                       decoder_input_ids=decoder_input_ids,
                                       labels=labels,
                                       struct_attention=struct_attention,
                                       relative_position=linear_relative_position_matrix)
            elif self.matrix == "all":
                model_outputs = self.model(input_ids=input_ids,
                                       attention_mask=enc_attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       labels=labels,
                                       struct_attention=struct_attention,
                                       relative_position=linear_relative_position_matrix)


        else:
            model_outputs = self.model(input_ids=input_ids,
                                       attention_mask=enc_attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       labels=labels)

        return model_outputs

    def report_check(self, epoch_steps, ):
        if self.enable_epoch_training:
            if epoch_steps and epoch_steps % self.report_every == 0:
                return True

    def train_one_epoch(self):

        # pbar = tqdm(self.train_data_loader)
        self.model.train()

        losses = 0.0
        # total_steps = len(self.train_dataloader) // self.update_freq if not self.enable_iterable_dataset else self.max_steps
        epoch_steps = 0

        total_losses = 0.0

        updated_step = 0
        for batch in self.train_dataloader:
            # pbar.set_description("Epoch {}, Step: {}".format(epoch, step))
            model_outputs = self.step(batch)

            ce_loss = model_outputs[0]

            # ce_loss = ce_loss / self.update_freq

            assert not torch.isnan(ce_loss), "loss is NAN at Epoch: {} / Step: {}".format(self.trained_epochs,
                                                                                          self.trained_steps)

            losses += ce_loss.item()
            total_losses += ce_loss.item()

            ce_loss.backward()

            updated_step += 1
            if updated_step == self.update_freq:
                """
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                # self.model.zero_grad()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                """
                self.model_update()
                updated_step = 0

                self.trained_steps += 1
                if self.enable_iterable_dataset:
                    self.update_epoch_for_iterable_dataset()

                epoch_steps += 1

                if epoch_steps and epoch_steps % self.report_every == 0:
                    avg_loss = losses / self.report_every
                    # avg_loss = losses
                    losses = 0.0
                    if self.lr_scheduler is not None:
                        lr = "%.8f" % self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.learning_rate
                    spend_time = "%.4f" % (time.time() - self.start_time)
                    if self.max_epochs > 0:
                        aim_information = "Total {} {}, Cur:".format(self.max_epochs, 'epochs')
                    else:
                        aim_information = "Total {} {}, Cur:".format(self.max_steps, 'steps')

                    self.logger.info("{} epoch {} / step {}, finished: {}/{}, loss: {}, "
                                     "lr: {}, rank: {} spending {} s".format(aim_information,
                                                                             self.trained_epochs,
                                                                             self.trained_steps,
                                                                             epoch_steps,
                                                                             self.steps_one_epoch,
                                                                             '%.4f' % avg_loss,
                                                                             lr,
                                                                             self.rank,
                                                                             spend_time))

                # training according step
                if not self.enable_epoch_training:
                    is_stopping = self.evaluate()
                    if is_stopping or self.training_stop_check():  # early stopping
                        return True

        if updated_step > 0:
            self.model_update()
            self.trained_steps += 1

            if not self.enable_epoch_training:
                is_stopping = self.evaluate()
                if is_stopping or self.training_stop_check():  # early stopping
                    return True

        self.trained_epochs += 1

        return False
        # return total_losses / total_steps

    def model_update(self):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        # self.model.zero_grad()
        self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def evaluate(self):
        """

        :return: if true: early stopping, stopping training
        """
        if self.evaluate_check():

            # prefix = "epoch_{}_step_{}_loss_{}".format(self.trained_epochs, self.trained_steps, "%.4f" % loss)
            prefix = "epoch_{}_step_{}".format(self.trained_epochs, self.trained_steps)
            if self.dist_train:
                model = self.model.module
            else:
                model = self.model
            metrics, output_file_name = self.evaluator.evaluate(model=model, device=self.device, args = self.args, prefix=prefix)

            saved_model_file_name = output_file_name.replace('.txt', '.pkl')

            tmp_result = dict()
            tmp_result['steps'] = self.trained_steps
            # tmp_result['loss'] = losses
            tmp_result['metrics'] = metrics
            self.evaluate_metric_results[
                'epoch-{}-steps-{}'.format(self.trained_epochs, self.trained_steps)] = tmp_result
            # self.metric_file_fin.write()

            val_metric = metrics[self.val_metric]

            obleu = metrics['officle_bleu']
            if obleu > self.best_obleu:
                self.best_obleu = obleu

            if val_metric > self.best_val_metric:
                self.best_epoch = self.trained_epochs
                self.best_step = self.trained_steps
                self.best_val_metric = val_metric
                self.save(os.path.join(self.model_saved_dir, 'best.pkl'))

            if self.max_keep_checkpoints == -1 or len(self.keep_checkpoints) < self.max_keep_checkpoints:
                self.keep_checkpoints.append((saved_model_file_name, val_metric))
                self.save(os.path.join(self.model_saved_dir, saved_model_file_name))
            else:
                self.keep_checkpoints = self.sort(self.keep_checkpoints)
                if self.keep_checkpoints[-1][1] < val_metric:
                    self.delete(os.path.join(self.model_saved_dir, self.keep_checkpoints[-1][0]))
                    self.keep_checkpoints[-1] = (saved_model_file_name, val_metric)
                    # keep_checkpoints.append((saved_model_file_name, val_metric))
                    self.save(os.path.join(self.model_saved_dir, saved_model_file_name))

            self.logger.info("Evaluating results at Epoch {}/ Step {}, {}: {}, avg_length: {}, "
                             "best metric is {} at epoch {}, steps: {}, "
                             "official bleu is {},"
                             "{} epochs and {} steps no improved".format(self.trained_epochs,
                                                                         self.trained_steps,
                                                                         self.val_metric,
                                                                         val_metric,
                                                                         metrics['avg_length'],
                                                                         self.best_val_metric,
                                                                         self.best_epoch,
                                                                         self.best_step,
                                                                         self.best_obleu,
                                                                         self.trained_epochs - self.best_epoch,
                                                                         self.trained_steps - self.best_step))

            is_early_stopping = self.early_stopping_check()
            self.model.train()
        else:
            is_early_stopping = False

        if self.dist_train:
            torch_dist.barrier()
        return is_early_stopping

        # return metrics, output_file_name

    def early_stopping_check(self):
        return False
        if self.early_stopping_patience > 0:
            if self.enable_epoch_training:
                if self.trained_epochs - self.best_epoch >= self.early_stopping_patience:
                    return True
            else:
                if self.trained_steps - self.best_step >= self.early_stopping_patience:
                    return True
        return False

    def save(self, saved_model_path):
        if not self.dist_train:
            save_flag = True
        else:
            save_flag = self.rank == 0

        if save_flag:
            model = self.model.module if self.dist_train else self.model
            torch.save(
                {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                 for k, v in model.state_dict().items()}, saved_model_path)

    @staticmethod
    def delete(file_src):
        # os.rmdir(file_src)
        os.remove(file_src)

    @property
    def max_steps(self):
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        # num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        # effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices

        max_steps = self.args.max_steps
        if max_steps < 0:
            max_steps = len(self.train_dataloader) * self.max_epochs
        return max_steps

    def update_epoch_for_iterable_dataset(self):
        self.trained_epochs = self.trained_steps // self.steps_one_epoch

    def obtrain_steps_for_one_epoch(self):
        if self.enable_iterable_dataset:
            steps_one_epoch = math.ceil(len(self.train_dataloader) / self.update_freq / self.word_size)
        else:
            steps_one_epoch = math.ceil(len(self.train_dataloader) / self.update_freq)

        return steps_one_epoch
