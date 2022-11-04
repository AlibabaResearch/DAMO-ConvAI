# -*- coding:utf-8 -*-

import datetime
import hashlib
import math
import json
import os
import shutil
import time

import codecs
import numpy as np
import pickle
import tensorflow as tf

from common import dump_script
from data import data_provider
from config import get_parser


def average_gradients(tower_grads):
    """
    多卡梯度求平均
    """
    average_grad = []
    variable_name = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        if grads:
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
        else:
            grad = None

        v = grad_and_vars[0][1]
        average_grad.append(grad)
        variable_name.append(v)
    return average_grad, variable_name


class Wrapper(object):
    """
    模型包装器
    """
    def __init__(self, args):
        self.data_provider = None

        # 训练需要的GPU
        self.train_gpu = []
        self.test_gpu = []
        self.train_gpu_num = 0
        self.test_gpu_num = 0

        # 是否采用多答案模型
        self.train_multi_mode = False
        self.test_multi_mode = False

        # 最大准确率
        self.max_test_acc = 0.0
        self.saver = None
        self.args = args

    def clean_model_dir(self):
        """
        清除model文件夹
        """
        if os.path.exists(self.args.model_dir):
            shutil.rmtree(self.args.model_dir)
        os.makedirs(self.args.model_dir)

    def init_data_source(self):
        """
        初始化文本版本数据源
        """
        self.train_gpu = self.args.train_gpu.split(",")
        self.test_gpu = self.args.test_gpu.split(",")
        self.train_gpu_num = len(self.train_gpu)
        self.test_gpu_num = len(self.test_gpu)

        self.train_multi_mode = self.args.train_multi_mode
        self.test_multi_mode = self.args.test_multi_mode

        if (self.train_multi_mode is True and self.args.train_response_num < 2) or (
            self.test_multi_mode is True and self.args.test_response_num < 2):
            raise RuntimeError("response_numer must equal or larger than 2 under multi_mode")
        elif (self.train_multi_mode is False and self.args.train_pack_label is True) or (
            self.test_multi_mode is False and self.args.test_pack_label is True):
            raise RuntimeError("pack label is disallowed under single mode")

        self.data_provider = data_provider.DataProvider(do_lower=self.args.do_lower, min_df=self.args.min_df,
            line_sep_char=self.args.line_sep_char, turn_sep_char=self.args.turn_sep_char,
            token_sep_char=self.args.token_sep_char)
        self.data_provider.init_gpu_num(self.train_gpu_num, self.test_gpu_num)
        self.data_provider.init_data_source(self.args.column_num, self.args.train_file, self.args.test_file,
            self.args.train_pack_label, self.args.test_pack_label, self.args.train_response_num,
            self.args.test_response_num, self.args.round_num, self.args.max_seq_len, self.args.class_num)

    def prepare_data_dict(self):
        """
        读取预训练的向量并更新词典
        """
        if self.args.pretrain_embedding_path:
            self.data_provider.init_w2v_matrix(self.args.pretrain_embedding_path)
        else:
            print("no pretrain embedding, build from source")
            self.data_provider.make_corpus()

        self.vocab_id_map = self.data_provider.vocab_id_map
        self.id_vocab_map = self.data_provider.id_vocab_map
        self.vocab_size = self.data_provider.vocab_size
        self.w2v_matrix = self.data_provider.w2v_matrix

    def dump_environment(self):
        """
        dump环境配置（包括模型参数及字典）到文件, 这里的文件是相对路径，对于文本数据源
        为保证字典和模型的一致性，这里必须清理目录，删除包括模型文件的所有文件
        """
        self.clean_model_dir()  # 更新文件夹
        data_file = os.path.join(self.args.model_dir, "%s.pkl" % self.args.env_name)

        data_dict = self.data_provider.dump_dictionary()
        data_dict["environment"] = self.args
        with open(data_file, "wb") as f_out:
            pickle.dump(data_dict, f_out)

    def load_environment(self, model_dir=None, env_name=None):
        """
        加载环境配置，这里的文件是相对路径
        """
        model_dir = self.args.model_dir if model_dir is None else model_dir
        env_name = self.args.env_name if env_name is None else env_name

        data_file = os.path.join(model_dir, "%s.pkl" % env_name)
        with codecs.open(data_file, "rb") as f_in:
            data_dict = pickle.load(f_in)

        self.args = data_dict["environment"]
        self.args.model_dir = model_dir
        self.args.env_name = env_name

        self.init_data_source()
        self.data_provider.load_dictionary(data_dict)

        self.vocab_id_map = self.data_provider.vocab_id_map
        self.id_vocab_map = self.data_provider.id_vocab_map
        self.vocab_size = self.data_provider.vocab_size

    def init_data_socket(self):
        """
        设置data set接入
        """
        self.dt_types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
            tf.int32, tf.int32, tf.int32, tf.float32)
        shape_list = [tf.TensorShape([None] * (i + 1)) for i in range(3)]
        self.dt_shapes = tuple([shape_list[i - 1] for i in (3, 2, 1, 3, 2, 1, 3, 2, 2, 1, 2)])

    def get_train_handle(self, session):
        """
        训练过程初始化
        """
        train_func = lambda: self.data_provider.train_data_generator(load_batch=self.args.load_batch,
            partition_num=self.args.partition_num)
        train_dataset = tf.data.Dataset.from_generator(train_func, self.dt_types, self.dt_shapes).prefetch(
            self.args.train_batch_size * 100)
        train_dataset = train_dataset.batch(self.args.train_batch_size)
        train_dataset = train_dataset.repeat(self.args.epoches)

        # op for session run to get handle string
        train_handle_op = train_dataset.make_one_shot_iterator().string_handle()
        train_handle = session.run(train_handle_op)
        return train_handle

    def get_test_handle(self, session):
        """
        测试过程初始化
        """
        test_func = lambda: self.data_provider.test_data_generator(partition_num=self.args.partition_num)
        test_dataset = tf.data.Dataset.from_generator(test_func, self.dt_types, self.dt_shapes).prefetch(
            self.args.train_batch_size * 100)
        test_dataset = test_dataset.batch(self.args.test_batch_size)

        # op for session run to get handle string
        test_handle_op = test_dataset.make_one_shot_iterator().string_handle()
        test_handle = session.run(test_handle_op)
        return test_handle

    def create_optimizer(self):
        """
        建立模型公用的优化器
        """
        self.global_step = tf.get_variable("step", dtype=tf.int32, initializer=tf.constant(0))
        self.lr = tf.get_variable("learning_rate", dtype=tf.float32, initializer=tf.constant(self.args.lr))

        global_step = tf.cast(self.global_step, dtype=tf.float32)
        lr_next = tf.reduce_max([self.args.lr * tf.pow(self.args.lr_decay_rate,
            tf.cast(global_step / self.args.lr_decay_steps, dtype=tf.float32)), self.args.min_lr])

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(self.lr, lr_next, name="update_lr"))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.args.beta1, beta2=self.args.beta2)
        return optimizer

    def build_graph(self, is_train):
        """
        建立模型图
        estimation: batch, answer, stack, 2
        score: batch, answer, 2
        label: batch, answer
        """
        if is_train is False:
            contexts_shape = [None, None, self.args.round_num, self.args.max_seq_len]
            context_lens_shape = [None, None, self.args.round_num]
            rounds_shape = [None, None]
            responses_shape = [None, None, self.args.test_response_num, self.args.max_seq_len]
            response_lens_shape = [None, None, self.args.test_response_num]
            roles_shape = [None, None, None]
            data_types_shape = [None, None]
            labels_shape = [None, None, self.args.test_response_num]

            self.contexts = tf.placeholder(tf.int32, contexts_shape)
            self.context_lens = tf.placeholder(tf.int32, context_lens_shape)
            self.rounds = tf.placeholder(tf.int32, rounds_shape)
            self.next_contexts = tf.placeholder(tf.int32, contexts_shape)
            self.next_context_lens = tf.placeholder(tf.int32, context_lens_shape)
            self.next_rounds = tf.placeholder(tf.int32, rounds_shape)
            self.responses = tf.placeholder(tf.int32, responses_shape)
            self.response_lens = tf.placeholder(tf.int32, response_lens_shape)
            self.roles = tf.placeholder(tf.int32, roles_shape)
            self.data_types = tf.placeholder(tf.int32, data_types_shape)
            self.labels = tf.placeholder(tf.float32, labels_shape)
        else:
            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(self.handle, self.dt_types)
            next_element = iterator.get_next()

            self.contexts = next_element[0]
            self.context_lens = next_element[1]
            self.rounds = next_element[2]
            self.next_contexts = next_element[3]
            self.next_context_lens = next_element[4]
            self.next_rounds = next_element[5]
            self.responses = next_element[6]
            self.response_lens = next_element[7]
            self.roles = next_element[8]
            self.data_types = next_element[9]
            self.labels = next_element[10]

            """ # for debug
            contexts_shape = [1, 1, self.args.round_num, self.args.max_seq_len]
            context_lens_shape = [1, 1, self.args.round_num]
            rounds_shape = [1, 1]
            responses_shape = [1, 1, self.args.test_response_num, self.args.max_seq_len]
            response_lens_shape = [1, 1, self.args.test_response_num]
            labels_shape = [1, 1, self.args.test_response_num]

            self.contexts = tf.reshape(self.contexts, contexts_shape)
            self.context_lens = tf.reshape(self.context_lens, context_lens_shape)
            self.rounds = tf.reshape(self.rounds, rounds_shape)
            self.next_contexts = tf.reshape(self.next_contexts, contexts_shape)
            self.next_context_lens = tf.reshape(self.next_context_lens, context_lens_shape)
            self.next_rounds = tf.reshape(self.next_rounds, rounds_shape)
            self.responses = tf.reshape(self.responses, responses_shape)
            self.response_lens = tf.reshape(self.response_lens, response_lens_shape)
            self.labels = tf.reshape(self.labels, labels_shape)
            """

        # 建立模型图
        model_params = {
            "vocab_num": self.vocab_size,
            "round_num": self.args.round_num,
            "max_seq_len": self.args.max_seq_len,
            "embedding_size": self.args.embedding_size,
            "hidden_size": self.args.hidden_size,
            "kernel_size": self.args.kernel_size,
            "filter_num": self.args.filter_num,
            "block_num": self.args.block_num,
            "head_num": self.args.head_num,
            "enc_num": self.args.enc_num,
            "class_num": self.args.class_num,
            "trainable_embedding": self.args.trainable_embedding,
            "bidirectional": self.args.bidirectional,
            "l2_reg": self.args.l2_reg
        }

        print("unsupported model type, please check")
        exit(0)

        optimizer = self.create_optimizer()

        # 构建训练图, 不要在模型中使用Variable创建变量，避免多设备不一致
        # tower_train_label不能直接使self.label，由于其第一维和第二维是反置的
        if is_train:
            tower_grad, tower_loss, tower_train_score, tower_train_label = [], [], [], []
            tower_train_context_feature, tower_train_response_feature = [], []
            for data_index, gpu_index in zip(range(self.train_gpu_num), self.train_gpu):
                contexts = tf.reshape(self.contexts[:, data_index, :, :], [-1, self.args.round_num, self.args.max_seq_len])
                context_lens = tf.reshape(self.context_lens[:, data_index, :], [-1, self.args.round_num])
                rounds = tf.reshape(self.rounds[:, data_index], [-1])
                next_contexts = tf.reshape(self.next_contexts[:, data_index, :, :],
                    [-1, self.args.round_num, self.args.max_seq_len])
                next_context_lens = tf.reshape(self.next_context_lens[:, data_index, :], [-1, self.args.round_num])
                next_rounds = tf.reshape(self.next_rounds[:, data_index], [-1])
                responses = tf.reshape(self.responses[:, data_index, :, :], [-1, self.args.train_response_num, self.args.max_seq_len])
                response_lens = tf.reshape(self.response_lens[:, data_index, :], [-1, self.args.train_response_num])
                roles = tf.reshape(self.roles[:, data_index, :], [-1, self.args.round_num * 2 + 1])
                data_types = tf.reshape(self.data_types[:, data_index], [-1])
                labels = tf.reshape(self.labels[:, data_index, :], [-1, self.args.train_response_num])
                with tf.device("/gpu:{}".format(gpu_index)):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as _:
                        tf_op = self.model.forward(contexts, context_lens, rounds, next_contexts, next_context_lens,
                            next_rounds, responses, response_lens, roles, data_types, self.args.train_response_num,
                            self.args.keep_prob)
                    loss_func = self.model.calculate_loss if self.train_multi_mode else self.model.calculate_loss
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        loss = loss_func(tf_op["estimation"], labels)
                        # loss = loss_func(tf_op["estimation"], labels) + 1.0 * tf_op["extra_loss"]
                    grad = optimizer.compute_gradients(loss)
                    tower_grad.append(grad)
                    tower_loss.append(loss)
                    tower_train_score.append(tf_op["score"])
                    tower_train_label.append(labels)
                    tower_train_context_feature.append(tf_op["context_feature"])
                    tower_train_response_feature.append(tf_op["response_feature"])

            grads, variables = average_gradients(tower_grad)
            grads, gnorm = tf.clip_by_global_norm(grads, 3.0)
            self.gnorm = gnorm
            self.train_op = optimizer.apply_gradients(list(zip(grads, variables)), global_step=self.global_step)

            self.loss = sum(tower_loss) / self.train_gpu_num
            self.train_score = tower_train_score
            self.train_label = tower_train_label

        # 构建测试图用
        tower_attn_test_score, tower_test_score, tower_test_label = [], [], []
        tower_test_context_feature, tower_test_response_feature = [], []
        for data_index, gpu_index in zip(range(self.test_gpu_num), self.test_gpu):
            contexts = tf.reshape(self.contexts[:, data_index, :, :], [-1, self.args.round_num, self.args.max_seq_len])
            context_lens = tf.reshape(self.context_lens[:, data_index, :], [-1, self.args.round_num])
            rounds = tf.reshape(self.rounds[:, data_index], [-1])
            next_contexts = tf.reshape(self.next_contexts[:, data_index, :, :],
                [-1, self.args.round_num, self.args.max_seq_len])
            next_context_lens = tf.reshape(self.next_context_lens[:, data_index, :], [-1, self.args.round_num])
            next_rounds = tf.reshape(self.next_rounds[:, data_index], [-1])
            responses = tf.reshape(self.responses[:, data_index, :, :], [-1, self.args.test_response_num, self.args.max_seq_len])
            response_lens = tf.reshape(self.response_lens[:, data_index, :], [-1, self.args.test_response_num])
            roles = tf.reshape(self.roles[:, data_index, :], [-1, self.args.round_num * 2 + 1])
            data_types = tf.reshape(self.data_types[:, data_index], [-1])
            labels = tf.reshape(self.labels[:, data_index, :], [-1, self.args.test_response_num])
            with tf.device("/gpu:{}".format(gpu_index)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as _:
                    tf_op = self.model.forward(contexts, context_lens, rounds, next_contexts, next_context_lens,
                        next_rounds, responses, response_lens, roles, data_types, self.args.test_response_num, 1.0)
                tower_attn_test_score.append(tf_op["attention_score"])
                tower_test_score.append(tf_op["score"])
                tower_test_label.append(labels)
                tower_test_context_feature.append(tf_op["context_feature"])
                tower_test_response_feature.append(tf_op["response_feature"])

        self.attn_test_score = tower_attn_test_score
        self.test_score = tower_test_score
        self.test_label = tower_test_label
        self.test_context_feature = tower_test_context_feature
        self.test_response_feature = tower_test_response_feature

    def calc_metrics(self, scores, labels, multi_mode):
        """
        计算指标, preds&labels shape: [gpu_number, batch_size]
        """
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)

        if multi_mode:
            #scores = np.sum(scores, axis=2)[:, :, 1]  # [batch, answer_num, 2] hard code, dimension 2 pos 1(score 1)
            scores = scores[:, :, 1, 1]  # [batch, answer_num, 2] hard code, dimension 2 pos 1(score 1)
            argsort_labels = np.argsort(-labels, axis=-1)
        else:
            scores = scores[:, 0, :]  # hard code, dimension 1 pos 0(answer 0), label must be [batch_size, 2]
            argsort_labels = labels

        argsort_scores = np.argsort(-scores, axis=-1)

        accuracy = np.mean(np.equal(argsort_scores[:, 0:1], argsort_labels[:, 0:1]))
        accuracy2 = np.mean(np.sum(np.equal(argsort_scores[:, 0:2], argsort_labels[:, 0:1]), axis=-1))
        accuracy5 = np.mean(np.sum(np.equal(argsort_scores[:, 0:5], argsort_labels[:, 0:1]), axis=-1))

        accuracy, accuracy2, accuracy5 = float(accuracy), float(accuracy2), float(accuracy5)
        return accuracy, accuracy2, accuracy5

    def train_procedure(self, session, is_continue_train=False):
        """
        训练模型
        """
        if is_continue_train is False:
            session.run(tf.global_variables_initializer())

        if self.args.pretrain_embedding_path:
            embedding_matrix = self.data_provider.w2v_matrix
            if is_continue_train is False:
                self.model.set_embedding_matrix(session, embedding_matrix)
        else:
            print("no pretrain embedding used.")

        train_handle = self.get_train_handle(session)
        run_list = [self.train_op, self.loss, self.lr, self.global_step, self.train_score, self.train_label]

        while True:
            try:
                _, loss, lr, step, scores, labels = session.run(run_list, feed_dict={self.handle: train_handle})
                if step % self.args.print_per_steps == 0:
                    acc, acc2, acc5 = self.calc_metrics(scores, labels, self.train_multi_mode)
                    print("step: %s, ls: %.3f, lr: %.5f, acc: %.3f, acc2: %.3f, acc5: %.3f" % (
                        step, loss, lr, acc, acc2, acc5))
                if step % self.args.eval_per_steps == 0:
                    self.test_procedure(session, step)
            except tf.errors.OutOfRangeError as _:
                print("run out of range, epoch done!")
                break
        self.test_procedure(session, step)

    def test_procedure(self, session, step):
        """
        测试模型
        """
        test_handle = self.get_test_handle(session)
        final_step, total_scores, total_labels = 0, [], []
        run_list = [self.test_score, self.test_label]
        while True:
            try:
                scores, labels = session.run(run_list, feed_dict={self.handle: test_handle})
                #for score in scores:
                #    print(score[:, :, 0, 1])
                #    input()

                total_scores.extend(scores)
                total_labels.extend(labels)
                final_step += 1
            except tf.errors.OutOfRangeError as _:
                print("test procedure finished at %d steps" % final_step)
                break
        # 多step维度折平
        assert (len(total_scores) == len(total_labels))
        acc, acc2, acc5 = self.calc_metrics(total_scores, total_labels, self.test_multi_mode)

        if acc > self.max_test_acc:
            self.max_test_acc = acc
        self.save_model(session, step)
        print("test_acc: %.3f, acc_2: %.3f, acc_5: %.3f, best acc is: %.3f" % (acc, acc2, acc5, self.max_test_acc))

    def save_model(self, session, step):
        """
        保存模型
        """
        if not self.saver:
            self.saver = tf.train.Saver(max_to_keep=100)

        model_file = os.path.join(self.args.model_dir, self.args.model_name + ".ckpt." + str(step))
        self.saver.save(session, model_file)
        return True

    def load_model(self, session, ckpt_step):
        """
        加载模型
        """
        if not self.saver:
            self.saver = tf.train.Saver()
        self.saver.restore(session, os.path.join(self.args.model_dir, self.args.model_name + ".ckpt.%s" % ckpt_step))
        return True

    def export_model(self, session, signature_name, serving_dir):
        """
        导出tf serving模式的model
        """
        if os.path.exists(serving_dir):
            shutil.rmtree(serving_dir)

        print("transform done!!")


def main():
    # command demo
    # python wrapper.py --data_dir /data/pipeishuju/ --model_dir /data/re2_model --model_type re2
    args = get_parser()
    print(json.dumps(vars(args), indent=4))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    np.set_printoptions(suppress=True,threshold=1000000)

    if args.stage == "train":
        with tf.Session(config=config) as session:
            wrapper = Wrapper(args)
            wrapper.init_data_socket()
            wrapper.init_data_source()
            wrapper.prepare_data_dict()
            wrapper.dump_environment()
            wrapper.build_graph(is_train=True)
            wrapper.train_procedure(session)
    elif args.stage == "continue_train":
        with tf.Session(config=config) as session:
            wrapper = Wrapper(args)
            wrapper.init_data_socket()
            wrapper.load_environment()
            wrapper.build_graph(is_train=True)
            wrapper.load_model(session, args.load_step)
            wrapper.train_procedure(session, is_continue_train=True)
    elif args.stage == "test":
        with tf.Session(config=config) as session:
            wrapper = Wrapper(args)
            wrapper.init_data_socket()
            wrapper.load_environment()
            wrapper.build_graph(is_train=True)
            wrapper.load_model(session, args.load_step)
            wrapper.test_procedure(session, step=0)
    elif args.stage == "export":
        with tf.Session(config=config) as session:
            wrapper = Wrapper(args)
            wrapper.init_data_socket()
            wrapper.load_environment()
            wrapper.build_graph(is_train=False)
            wrapper.export_model(session, args.signature_name, "to_add")


if __name__ == "__main__":
    main()
