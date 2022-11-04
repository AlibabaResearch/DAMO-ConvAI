#!/usr/bin/python
# _*_coding:utf-8_*_

import os
import codecs
import argparse
import logging
from typing import List
import pickle
from tqdm import tqdm

from optimization import BERTAdam
from data import data_provider
from network import Dial2vec
from metrics import *
from utils import split_matrix


class WrapperBert:
    """
    Dial2vec模型训练包装器
    """
    def __init__(self, args):
        self.args = args

        self.data_provider = None
        self.model = None
        self.max_acc = 0.0
        self.best_dev_evaluation_result = EvaluationResult()
        self.best_test_evaluation_result = EvaluationResult()
        self.best_epoch = -1

        self.logger = args.logger
        self.disable_tqdm = False if args.local_rank in [-1, 0] else True

    def init_data_socket(self):
        """
        初始化数据接口
        """
        # os.makedirs(self.args.output_dir, exist_ok=True)
        self.data_provider = data_provider.DataProvider(self.args)
        self.data_provider.init_data_socket()

    def load_model(self, init_checkpoint):
        """
        加载模型
        """
        self.args.num_labels = len(self.data_provider.get_labels())
        self.args.total_steps = self.data_provider.peek_num_train_examples()
        self.args.sep_token_id = self.data_provider.get_tokenizer().convert_tokens_to_ids(self.args.sep_token)

        self.model = Dial2vec(self.args)
        self.init_bert(init_checkpoint)
        self.model.set_finetune()
        self.model = self.model.to(self.args.device)

    def cosine_similarity(self, x, y):
        """
        计算向量的cosine值
        """
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def normalize_features(self, batch_feature):
        """
        对编码的特征做归一化
        """
        batch_feature = np.array(batch_feature, dtype=np.float32)
        batch_feature = batch_feature / (np.sqrt((batch_feature ** 2).sum(axis=1))[:, None] + 1e-6)
        return batch_feature

    def eval_tasks(
        self,
        strategy: str = 'mean_by_role',
        role: str = 'all',
        tasks: List[str] = None,
        mode: str = 'test',
        force: bool = False,
        features: np.array = None
    ):
        """
        集成评估API
        :param strategy:    str             embedding strategy
                                            [Options: mean: mean,
                                                      mean_by_role: mean w.r.t. role first, then add them together.]
        :param role:        str             Using the features from which interlocutor
        :param task:        List[str]
        :param mode:        str
        :param force:       bool            If force=True, we evaluate all metrics.
        :return:
        """
        if tasks is None:
            return

        if features is None:
            self.model.eval()
            test_loader = self.data_provider.get_clustering_test_loader(mode=mode)

            features = []
            with torch.no_grad():
                for step, batch in enumerate(test_loader):
                    batch = tuple(t.to(self.args.device) for t in batch)
                    output_dict = self.model(batch, strategy=strategy)
                    role2feat = {'all': 'final_feature', 'p1': 'q_feature', 'p2': 'r_feature'}
                    feature = output_dict[role2feat[role]]
                    features.append(feature)
            features = torch.cat(features)
            cpu_features = features.cpu()
        else:
            cpu_features = torch.tensor(features)
            features = cpu_features.to(self.args.device)

        test_path = os.path.join(self.args.data_dir, "clustering_%s.tsv" % mode)
        with codecs.open(test_path, "r", "utf-8") as f:
            labels = [int(line.strip('\n').split("\t")[-1]) for line in f]

        best_evaluation_result = self.best_test_evaluation_result if mode == 'test' else self.best_dev_evaluation_result
        evaluation_result = EvaluationResult()
        if 'clustering' in tasks:
            n_average = max(3, 10 - features.shape[0] // 500)
            er = feature_based_evaluation_at_once(features=cpu_features,
                                                  labels=labels,
                                                  n_average=n_average,
                                                  tsne_visualization_output=None,
                                                  tasks=['clustering'],
                                                  dtype='float32',
                                                  logger=None,
                                                  note=','.join([mode, strategy, role]))
            evaluation_result.RI = er.RI
            evaluation_result.NMI = er.NMI
            evaluation_result.acc = er.acc
            evaluation_result.purity = er.purity

        is_best = True if evaluation_result > best_evaluation_result else False     # based on acc, plz ref metrics.py->EvaluationResult.__lr__()

        if 'semantic_relatedness' in tasks or 'session_retrieval' in tasks:
            if is_best or mode == 'test' or force:
                er = feature_based_evaluation_at_once(features=cpu_features,
                                                      labels=labels,
                                                      n_average=0,
                                                      tsne_visualization_output=None,
                                                      tasks=['semantic_relatedness', 'session_retrieval'],
                                                      dtype='float32',
                                                      logger=None,
                                                      note=','.join([mode, strategy, role]))

                evaluation_result.SR = er.SR
                evaluation_result.MRR = er.MRR
                evaluation_result.MAP = er.MAP

        if 'align_uniform' in tasks:
            if is_best or mode == 'test' or force:
                er = feature_based_evaluation_at_once(features=cpu_features,
                                                      labels=labels,
                                                      gpu_features=features,
                                                      n_average=0,
                                                      tsne_visualization_output=None,
                                                      tasks=['align_uniform'],
                                                      dtype='float32',
                                                      logger=None,
                                                      note=','.join([mode, strategy, role]))

                evaluation_result.alignment = er.alignment
                evaluation_result.adjusted_alignment = er.adjusted_alignment
                evaluation_result.uniformity = er.uniformity

        evaluation_result.show(logger=self.logger,
                               note=','.join([mode, strategy, role]))

        return is_best, evaluation_result

    def get_feature(self, mode, output_filename, strategy='mean_by_role', role='all'):
        """
        generate features for dialogues
        :param: mode
        :return:
        """
        self.model.eval()
        feature_loader = self.data_provider.get_clustering_test_loader(mode=mode, level='sentence')

        features, labels, guids = [], [], []
        with torch.no_grad():
            with tqdm(total=feature_loader.__len__() * self.args.test_batch_size, ncols=90, disable=self.disable_tqdm) as pbar:
                for step, batch in enumerate(feature_loader):
                    batch = tuple(t.to(self.args.device) for t in batch)
                    output_dict = self.model(batch, strategy=strategy)
                    role2feat = {'all': 'final_feature', 'p1': 'q_feature', 'p2': 'r_feature'}
                    feature = output_dict[role2feat[role]]
                    features.append(feature)
                    guids.append(batch[7][:, 0])
                    pbar.update(self.args.test_batch_size)

        features = torch.cat(features).detach().cpu().numpy()
        guids = torch.cat(guids).detach().cpu().numpy()

        splited_features = np.split(features, np.unique(guids, return_index=True)[1][1:])
        final_features = [np.mean(splited_feature, axis=0) for splited_feature in splited_features]
        final_features = np.stack(final_features)

        pickle.dump(obj={'features': features,
                         'guids': guids,
                         'final_features': final_features},
                    file=open(output_filename, 'wb'))

        self.logger.info('Save features to -> [%s]' % output_filename)

    def train(self):
        """
        训练Dial2vec模型
        """
        self.logger.info("device: %s n_gpu: %s" % (self.args.device, self.args.n_gpu))

        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

        global_step = 0
        num_train_steps = self.data_provider.peek_num_train_examples()
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]
        optimizer = BERTAdam(optimizer_grouped_parameters, lr=self.args.learning_rate, warmup=self.args.warmup_proportion, t_total=num_train_steps)

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')

            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.fp16_opt_level)

        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.args.local_rank],
                                                                   output_device=self.args.local_rank,
                                                                   find_unused_parameters=False)   # 如果没有不参与计算的权重，find_unused_parameters=False可以提升运算速度。

        # 清空evaluation result cache, 避免vanilla backbone的影响
        self.best_test_evaluation_result = EvaluationResult()

        # 启动训练模式
        self.model.train()

        train_loader = self.data_provider.get_train_loader()
        for epoch in range(int(self.args.num_train_epochs)):
            with tqdm(total=train_loader.__len__() * self.args.train_batch_size, ncols=90, disable=self.disable_tqdm) as pbar:
                for step, batch in enumerate(train_loader):
                    batch = tuple(t.to(self.args.device) for t in batch)
                    output_dict = self.model(batch, strategy='mean_by_role')
                    loss = output_dict['loss']

                    if self.args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    optimizer.zero_grad()

                    if self.args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    # 梯度裁剪
                    # if self.args.fp16:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), -10, 10)
                    # else:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), -10, 10)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        global_step += 1

                    if global_step % self.args.print_interval == 0:
                        pbar.set_postfix(epoch=epoch, global_step=global_step, train_loss=float(loss.item()))

                    if global_step % self.args.test_interval == 0:
                        is_best, dev_evaluation_result = self.eval_tasks(strategy='mean_by_role',
                                                                         tasks=['clustering', 'semantic_relatedness', 'session_retrieval', 'align_uniform'],
                                                                         mode='dev',
                                                                         force=True)
                        if is_best:
                            self.best_epoch = epoch
                            _, test_evaluation_result = self.eval_tasks(strategy='mean_by_role',
                                                                        tasks=['clustering', 'semantic_relatedness', 'session_retrieval', 'align_uniform'],
                                                                        mode='test')
                            self.best_model_name = os.path.join(self.args.output_dir,
                                                      "%s.%s.%st.%sws.best_model.pkl" % (self.args.backbone,
                                                                                         self.args.dataset,
                                                                                         self.args.temperature,
                                                                                         self.args.max_turn_view_range))
                            if self.args.local_rank in [-1, 0]:
                                torch.save(self.model.module.state_dict(), self.best_model_name)
                                self.logger.info('Save best model to -> [%s] on LOCAL_RANK=%s' % (self.best_model_name, self.args.local_rank))
                            self.best_dev_evaluation_result = dev_evaluation_result
                            self.best_test_evaluation_result = test_evaluation_result

                        # 恢复训练模式
                        self.model.train()

                    pbar.update(self.args.train_batch_size)

        self.logger.info('=' * 10 + 'Final Best Testing Result: epoch=%s' % self.best_epoch + '=' * 10)
        self.best_test_evaluation_result.show(logger=logger, note='test,mean_by_role')

    def visualize_attention(self, visualization_image_output=None, visualization_html_output=None):
        self.model.eval()
        test_loader = self.data_provider.get_clustering_test_loader(mode='test')
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                batch = tuple(t.to(self.args.device) for t in batch)
                output_dict = self.model(batch,
                                         strategy='mean_by_role',
                                         output_attention=True)
                attention_weight = output_dict['attention']
                attention_weight = attention_weight[0].detach().cpu()
                break

            mask = (attention_weight == 0.).float()
            attention_weight = torch.softmax(attention_weight + mask * -1e9, dim=-1)
            attention_weight = attention_weight * (1 - mask)

            input_ids = batch[0][0, 0, :]
            role_ids = batch[3][0, 0, :]
            turn_ids = batch[4][0, 0, :]

            tokenizer = self.data_provider.get_tokenizer()
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            textual_show = '=' * 10 + 'Session' + '=' * 10 + '\n'
            pre_role = 'A' if role_ids[0] == 1 else 'B'
            textual_show += '%s: ' % pre_role

            for i, token in enumerate(input_tokens):
                if input_tokens[i] == '[PAD]':
                    break
                now_role = 'A' if role_ids[i] == 1 else 'B'

                if input_tokens[i].startswith('##'):
                    space_flag = False
                else:
                    space_flag = True

                if now_role == pre_role:
                    textual_show += input_tokens[i].lstrip('#')
                else:
                    textual_show += '\n%s: ' % now_role + input_tokens[i].lstrip('#')

                if space_flag:
                    textual_show += ' '

                pre_role = now_role

            textual_show += '\n' + '=' * 10 + 'End' + '=' * 10 + '\n'

            turn_lengths = []
            pre_turn_id, cache_turn_id = turn_ids[0], [turn_ids[0]]
            for i in range(1, len(turn_ids)):
                if pre_turn_id != 0 and turn_ids[i] == 0:   # meet [PAD]
                    break

                if turn_ids[i] == pre_turn_id:
                    cache_turn_id.append(turn_ids[i])
                else:
                    turn_lengths.append(len(cache_turn_id))
                    cache_turn_id = [turn_ids[i]]
                    pre_turn_id = turn_ids[i]
            if cache_turn_id:
                turn_lengths.append(len(cache_turn_id))
                cache_turn_id = []

            output_matrix = split_matrix(matrix=attention_weight[:sum(turn_lengths), :sum(turn_lengths)],
                                         lengths=turn_lengths,
                                         reduction='mean')

            eps = 1e-6
            mask = output_matrix < eps
            fig = plt.figure(figsize=(10, 10))
            axis = seaborn.heatmap(output_matrix.numpy(),
                                   vmin=output_matrix[output_matrix > eps].min().numpy(),
                                   vmax=output_matrix[output_matrix > eps].max().numpy(),
                                   cmap="PuBu",
                                   mask=mask.numpy(),
                                   annot=True,
                                   fmt='.2f')
            plt.xticks(ticks=np.arange(len(turn_lengths)) + 0.5, labels=np.arange(len(turn_lengths)))
            plt.yticks(ticks=np.arange(len(turn_lengths)) + 0.5, labels=np.arange(len(turn_lengths)))
            plt.savefig(visualization_image_output, bbox_inches="tight")
            plt.close()

            HTML = """<!DOCTYPE html>
                <html>
                <head> 
                <meta charset="utf-8"> 
                <title>Visualization</title> 
                </head>
                <body>

                <img border="0" src="%s" alt="Image">
                <h2>Case</h2>
                %s
                </body>
                </html>""" % (os.path.split(visualization_image_output)[-1], textual_show.replace('\n', ' <br> '))

            with open(visualization_html_output, 'w', encoding='utf-8') as writer:
                writer.write(HTML)

    def init_bert(self, init_checkpoint=None):
        """
        初始化BERT模型
        """
        if init_checkpoint is not None:
            state_dict = torch.load(self.args.init_checkpoint, map_location="cpu")
            if init_checkpoint.endswith('pt'):
                self.model.bert.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)

            self.logger.debug(self.model.module) if hasattr(self.model, 'module') else self.logger.debug(self.model)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--seed", default=42, type=int, help='random seed')
    parser.add_argument("--dataset", default="doc2dial", type=str, help="Options: [bitod, doc2dial, metalwoz, mwoz, selfdialog, sgd]")
    parser.add_argument("--init_checkpoint", default=None, type=str, help='checkpoint of initializing backbone.')
    parser.add_argument("--config_file", default="model/plato/config.json", type=str)
    parser.add_argument("--feature_checkpoint", default=None, type=str, help='checkpoint of extracted features.')

    # Training
    parser.add_argument("--stage", default='train', type=str)
    ## fp 16 training
    parser.add_argument("--fp16", action="store_true", help="float16 training")
    parser.add_argument("--fp16_opt_level", default="O1", type=str, help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']")
    ## DDP
    parser.add_argument("--local_rank", default=-1, type=int, help="local rank for DDP training.")
    ## Optimization
    parser.add_argument("--train_batch_size", default=5, type=int, help='Be careful when using multi-gpu.')
    parser.add_argument("--test_batch_size", default=10, type=int, help='Be careful when using multi-gpu.')
    parser.add_argument("--dev_batch_size", default=10, type=int, help='Be careful when using multi-gpu.')
    parser.add_argument("--num_train_epochs", default=15, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    ## interval
    parser.add_argument("--print_interval", default=20, type=int)
    parser.add_argument("--test_interval", default=100, type=int)

    # Model
    parser.add_argument("--backbone", default='bert', type=str, help='Options: [bert, plato, roberta, t5, todbert, blender]')
    parser.add_argument("--use_turn_embedding", default="False", type=str2bool)
    parser.add_argument("--use_role_embedding", default="False", type=str2bool)
    parser.add_argument("--use_response", default="False", type=str2bool)
    parser.add_argument("--temperature", default=1., type=float)

    # Data
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_turn_view_range", default=1000, type=int)
    parser.add_argument("--max_context_length", default=15, type=int)
    parser.add_argument("--sep_token", default="[SEP]", type=str)
    parser.add_argument("--use_sep_token", default="True", type=str2bool)
    args = parser.parse_args()

    # general configuration
    work_dir = os.path.dirname(os.path.realpath(__file__))
    print(work_dir)

    if args.init_checkpoint is not None:
        if args.init_checkpoint.endswith('pt'):
            args.init_checkpoint = os.path.join(work_dir, "model", args.init_checkpoint)
        elif args.init_checkpoint.endswith('pkl'):
            args.init_checkpoint = os.path.join(work_dir, "output", args.init_checkpoint)

    if args.feature_checkpoint is not None:
        if args.feature_checkpoint.endswith('pt'):
            args.feature_checkpoint = os.path.join(work_dir, "model", args.feature_checkpoint)
        elif args.feature_checkpoint.endswith('pkl'):
            args.feature_checkpoint = os.path.join(work_dir, "output", args.feature_checkpoint)

    args.config_file = os.path.join(work_dir, "model", args.config_file)
    args.data_dir = os.path.join(work_dir, "datasets/%s" % args.dataset)
    args.output_dir = os.path.join(work_dir, "output")   # dumb directory setting

    # logging configuration
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger = logging.getLogger(__name__)
    args.logger = logger

    # GPU configuration
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        # 每个进程根据自己 的local_rank来设置应该使用的GPU
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    args.logger.info(args)

    torch.set_printoptions(profile="full")
    wrapper = WrapperBert(args)
    wrapper.init_data_socket()
    wrapper.load_model(args.init_checkpoint)

    if args.stage == "train":
        pre_train_time = time()
        wrapper.eval_tasks(strategy='mean',
                           tasks=['clustering', 'semantic_relatedness', 'session_retrieval'],
                           mode='test',
                           force=True)
        wrapper.train()
        # wrapper.load_model(wrapper.best_model_name)
        # wrapper.visualize_attention(visualization_image_output='./figures/%s_%s_visualize_attention.png' % (args.backbone, args.dataset),
        #                             visualization_html_output='./figures/%s_%s_visualize_attention.html' % (args.backbone, args.dataset))
        logger.info('Total time costs: %s mins' % ((time() - pre_train_time) / 60))
    elif args.stage == "test":
        pre_test_time = time()
        wrapper.eval_tasks(strategy='mean_by_role',
                           role='all',
                           tasks=['clustering', 'semantic_relatedness', 'session_retrieval', 'align_uniform'],
                           mode='test')
        wrapper.eval_tasks(strategy='mean',
                           role='all',
                           tasks=['clustering', 'semantic_relatedness', 'session_retrieval'],
                           mode='test')
        wrapper.eval_tasks(strategy='mean_by_role',
                           role='p1',
                           tasks=['clustering', 'semantic_relatedness', 'session_retrieval'],
                           mode='test')
        wrapper.eval_tasks(strategy='mean_by_role',
                           role='p2',
                           tasks=['clustering', 'semantic_relatedness', 'session_retrieval'],
                           mode='test')
        logger.info('Total time costs: %s mins' % ((time() - pre_test_time) / 60))
    elif args.stage == 'embedding':
        pre_embedding_time = time()
        wrapper.get_feature(mode='test', output_filename=args.feature_checkpoint)
        logger.info('Total time costs: %s mins' % ((time() - pre_embedding_time) / 60))
    elif args.stage == 'eval_from_embedding':
        pre_embedding_time = time()
        data = pickle.load(open(args.feature_checkpoint, 'rb'))
        features = data['final_features']
        wrapper.eval_tasks(strategy='mean_by_role',
                           role='all',
                           tasks=['clustering', 'semantic_relatedness', 'session_retrieval'],
                           mode='test',
                           force=True,
                           features=features)
        logger.info('Total time costs: %s mins' % ((time() - pre_embedding_time) / 60))
