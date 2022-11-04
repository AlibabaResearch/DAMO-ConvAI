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

from data import data_provider_bert
from model import bert
from model import dse_cl_bert
from config_bert import get_parser
import wrapper


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


class BertWrapper(wrapper.Wrapper):
    """
    模型包装器
    """
    def __init__(self, args):
        super(BertWrapper, self).__init__(args)

    def clean_model_dir(self):
        """
        清除model文件夹
        """
        if os.path.exists(self.args.model_save_dir):
            shutil.rmtree(self.args.model_save_dir)
        os.makedirs(self.args.model_save_dir)

    def clean_serving_dir(self):
        """
        清除serving文件夹
        """
        if os.path.exists(self.args.serving_dir):
            shutil.rmtree(self.args.serving_dir)
        os.makedirs(self.args.serving_dir)

    def init_data_source(self):
        """
        初始化文本版本数据源, 覆盖基类代码
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

        self.data_provider = data_provider_bert.BertDataProvider(do_lower=self.args.do_lower, min_df=self.args.min_df,
            line_sep_char=self.args.line_sep_char, turn_sep_char=self.args.turn_sep_char)
        self.data_provider.init_gpu_num(self.train_gpu_num, self.test_gpu_num)
        self.data_provider.init_data_source(self.args.column_num, self.args.train_file, self.args.test_file,
            self.args.train_pack_label, self.args.test_pack_label, self.args.train_response_num,
            self.args.test_response_num, self.args.round_num, self.args.max_seq_len, self.args.class_num)

    def dump_environment(self):
        """
        dump环境配置（包括模型参数及字典）到文件, 这里的文件是相对路径，对于文本数据源
        为保证字典和模型的一致性，这里必须清理目录，删除包括模型文件的所有文件
        """
        self.clean_model_dir()  # 更新文件夹
        data_file = os.path.join(self.args.model_save_dir, "%s.pkl" % self.args.env_name)

        data_dict = self.data_provider.dump_dictionary()
        data_dict["environment"] = self.args
        with open(data_file, "wb") as f_out:
            pickle.dump(data_dict, f_out)

    def prepare_data_dict(self):
        """
        读取预训练的向量并更新词典
        """
        self.data_provider.init_bert_tokenizer(os.path.join(self.args.bert_init_dir, self.args.vocab_file))

    def load_environment(self, model_dir=None, env_name=None, is_continue_train=False):
        """
        加载环境配置，这里的文件是相对路径
        """
        model_dir = self.args.model_save_dir if model_dir is None else model_dir
        env_name = self.args.env_name if env_name is None else env_name

        if is_continue_train is True:
            train_file, test_file = self.args.train_file, self.args.test_file
            train_gpu, test_gpu = self.args.train_gpu, self.args.test_gpu
            lr, min_lr = self.args.lr, self.args.min_lr
            layer_number = self.args.layer_num
            round_number = self.args.round_num
            train_batch_size = self.args.train_batch_size
            test_batch_size = self.args.test_batch_size
            print_per_steps = self.args.print_per_steps
            eval_per_steps = self.args.eval_per_steps
            train_response_num = self.args.train_response_num
            test_response_num = self.args.test_response_num
            load_batch = self.args.load_batch
            partition_num = self.args.partition_num

        data_file = os.path.join(model_dir, "%s.pkl" % env_name)
        with open(data_file, "rb") as f_in:
            data_dict = pickle.load(f_in)

        self.args = data_dict["environment"]
        self.args.model_save_dir = model_dir
        self.args.env_name = env_name
        self.args.bert_init_dir = model_dir
        self.args.use_init_model = False  # 加载配置情况下，强制不使用预训练参数

        if is_continue_train is True:
            self.args.train_file, self.args.test_file = train_file, test_file
            self.args.train_gpu, self.args.test_gpu = train_gpu, test_gpu
            self.args.lr, self.args.min_lr = lr, min_lr
            self.args.layer_num = layer_number
            self.args.round_num = round_number
            self.args.train_batch_size = train_batch_size
            self.args.test_batch_size = test_batch_size
            self.args.print_per_steps = print_per_steps
            self.args.eval_per_steps = eval_per_steps
            self.args.train_response_num = train_response_num
            self.args.test_response_num = test_response_num
            self.args.load_batch = load_batch
            self.args.partition_num = partition_num

            data_file = os.path.join(self.args.model_save_dir, "%s.pkl" % env_name)
            with open(data_file, "wb") as f_out:
                pickle.dump(data_dict, f_out)

        self.init_data_source()
        self.data_provider.init_bert_tokenizer(os.path.join(self.args.bert_init_dir, self.args.vocab_file))
        self.data_provider.load_dictionary(data_dict)

    def init_data_socket(self):
        """
        设置data set接入
        """
        self.dt_types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
            tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32)
        shape_list = [tf.TensorShape([None] * (i + 1)) for i in range(3)]
        self.dt_shapes = tuple([shape_list[i - 1] for i in (3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 2, 1, 2)])

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

    def build_graph(self, is_train, is_continue_train=False):
        """
        建立模型图
        estimation: batch, answer, stack, 2
        score: batch, answer, 2
        label: batch, answer
        """
        if is_train is False:
            contexts_shape = [None, None, self.args.round_num, self.args.max_seq_len]
            rounds_shape = [None, None]
            next_contexts_shape = [None, None, self.args.round_num, self.args.max_seq_len]
            next_rounds_shape = [None, None]
            responses_shape = [None, None, self.args.test_response_num, self.args.max_seq_len]
            roles_shape = [None, None, None]
            data_types_shape = [None, None]
            labels_shape = [None, None, self.args.test_response_num]

            self.context_input_ids = tf.placeholder(tf.int32, contexts_shape)
            self.context_input_masks = tf.placeholder(tf.int32, contexts_shape)
            self.context_segment_ids = tf.placeholder(tf.int32, contexts_shape)
            self.rounds = tf.placeholder(tf.int32, rounds_shape)

            self.next_context_input_ids = tf.placeholder(tf.int32, next_contexts_shape)
            self.next_context_input_masks = tf.placeholder(tf.int32, next_contexts_shape)
            self.next_context_segment_ids = tf.placeholder(tf.int32, next_contexts_shape)
            self.next_rounds = tf.placeholder(tf.int32, next_rounds_shape)

            self.response_input_ids = tf.placeholder(tf.int32, responses_shape)
            self.response_input_masks = tf.placeholder(tf.int32, responses_shape)
            self.response_segment_ids = tf.placeholder(tf.int32, responses_shape)
            self.roles = tf.placeholder(tf.int32, roles_shape)
            self.data_types = tf.placeholder(tf.int32, data_types_shape)
            self.labels = tf.placeholder(tf.float32, labels_shape)
        else:
            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(self.handle, self.dt_types)
            next_element = iterator.get_next()

            self.context_input_ids = next_element[0]
            self.context_input_masks = next_element[1]
            self.context_segment_ids = next_element[2]
            self.rounds = next_element[3]

            self.next_context_input_ids = next_element[4]
            self.next_context_input_masks = next_element[5]
            self.next_context_segment_ids = next_element[6]
            self.next_rounds = next_element[7]

            self.response_input_ids = next_element[8]
            self.response_input_masks = next_element[9]
            self.response_segment_ids = next_element[10]
            self.roles = next_element[11]
            self.data_types = next_element[12]
            self.labels = next_element[13]

        # 建立模型图
        model_params = {
            "layer_num": self.args.layer_num,
            "round_num": self.args.round_num,
            "max_seq_len": self.args.max_seq_len,
            "hidden_size": self.args.hidden_size,
            "class_num": self.args.class_num,
            "bidirectional": self.args.bidirectional
        }

        bert_config = bert.BertConfig.from_json_file(os.path.join(self.args.bert_init_dir, self.args.bert_config_file))
        init_checkpoint = os.path.join(self.args.bert_init_dir, self.args.init_checkpoint + ".ckpt")

        model_params["bert_config"] = bert_config
        model_params["init_checkpoint"] = init_checkpoint # 对于continue train，由load_model完成参数加载

        if self.args.model_type == "cl_bert":
            self.model = dse_cl_bert.CLBERT(model_params)
        else:
            print("unsupported model type, please check")
            exit(0)

        # 建立优化器
        optimizer = self.create_optimizer()

        # 构建训练图, 不要在模型中使用Variable创建变量，避免多设备不一致
        # tower_train_label不能直接使self.label，由于其第一维和第二维是反置的
        if is_train is True:
            tower_grad, tower_loss, tower_train_score, tower_train_label = [], [], [], []
            tower_train_context_feature, tower_train_response_feature = [], []
            for data_index, gpu_index in zip(range(self.train_gpu_num), self.train_gpu):
                context_input_ids = self.context_input_ids[:, data_index, :, :]
                context_input_masks = self.context_input_masks[:, data_index, :, :]
                context_segment_ids = self.context_segment_ids[:, data_index, :, :]
                rounds = self.rounds[:, data_index]
                next_context_input_ids = self.next_context_input_ids[:, data_index, :, :]
                next_context_input_masks = self.next_context_input_masks[:, data_index, :, :]
                next_context_segment_ids = self.next_context_segment_ids[:, data_index, :, :]
                next_rounds = self.next_rounds[:, data_index]
                response_input_ids = self.response_input_ids[:, data_index, :, :]
                response_input_masks = self.response_input_masks[:, data_index, :, :]
                response_segment_ids = self.response_segment_ids[:, data_index, :, :]
                roles = tf.reshape(self.roles[:, data_index, :], [-1, self.args.round_num * 2 + 1])
                data_types = self.data_types[:, data_index]
                labels = self.labels[:, data_index, :]

                with tf.device("/gpu:{}".format(gpu_index)):
                    use_init_checkpoint = True if data_index == 0 else False
                    use_init_checkpoint = use_init_checkpoint if self.args.use_init_model is True else False
                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as _:
                        tf_op = self.model.forward(context_input_ids, context_input_masks, context_segment_ids,
                            rounds, next_context_input_ids, next_context_input_masks, next_context_segment_ids,
                            next_rounds, response_input_ids, response_input_masks, response_segment_ids,
                            roles, data_types, self.args.train_response_num, True, use_init_checkpoint)
                    loss_func = self.model.calculate_loss if self.train_multi_mode else self.model.calculate_loss
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):  # 考虑BN
                        loss = loss_func(tf_op["estimation"], labels)

                    variables = []
                    print("number of original variables: %s" % len(tf.trainable_variables()))
                    for var in tf.trainable_variables():
                        if var.name.find("bert") != -1:
                            for i in range(1, self.args.layer_num + 1):
                                if var.name.find("/layer_%s/" % (bert_config.num_hidden_layers - i)) != -1:  # 使得Layer11/2可训练
                                    variables.append(var)
                            else:
                                continue
                        variables.append(var)
                    print("number of trainable variables: %s" % len(variables))

                    grad = optimizer.compute_gradients(loss, variables)
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
            context_input_ids = self.context_input_ids[:, data_index, :, :]
            context_input_masks = self.context_input_masks[:, data_index, :, :]
            context_segment_ids = self.context_segment_ids[:, data_index, :, :]
            rounds = self.rounds[:, data_index]
            next_context_input_ids = self.next_context_input_ids[:, data_index, :, :]
            next_context_input_masks = self.next_context_input_masks[:, data_index, :, :]
            next_context_segment_ids = self.next_context_segment_ids[:, data_index, :, :]
            next_rounds = self.next_rounds[:, data_index]
            response_input_ids = self.response_input_ids[:, data_index, :, :]
            response_input_masks = self.response_input_masks[:, data_index, :, :]
            response_segment_ids = self.response_segment_ids[:, data_index, :, :]
            roles = tf.reshape(self.roles[:, data_index, :], [-1, self.args.round_num * 2 + 1])
            data_types = self.data_types[:, data_index]
            labels = self.labels[:, data_index, :]

            with tf.device("/gpu:{}".format(gpu_index)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as _:
                    tf_op = self.model.forward(context_input_ids, context_input_masks, context_segment_ids,
                        rounds, next_context_input_ids, next_context_input_masks, next_context_segment_ids,
                        next_rounds, response_input_ids, response_input_masks, response_segment_ids,
                        roles, data_types, self.args.test_response_num, False, False)
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
        if self.args.model_type.find("cl") != -1:
            return 0.0, 0.0, 0.0

        scores = np.concatenate(scores)
        labels = np.concatenate(labels)

        if multi_mode:
            scores = np.sum(scores, axis=2)[:, :, 1]  # [batch, answer_num, 2] hard code, dimension 2 pos 1(score 1)
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
        if is_continue_train is False:  # 第二条件写完全表明不存在False/True组合
            session.run(tf.global_variables_initializer())

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
            var_list = tf.all_variables()

            """
            # 对于BN的保存改造
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
            bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]

            if bn_moving_vars or bn_moving_vars:
                print("find BN variables!")
            var_list += bn_moving_vars
            """

            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=100)

            vocab_file = os.path.join(self.args.bert_init_dir, self.args.vocab_file)
            des_vocab_file = os.path.join(self.args.model_save_dir, self.args.vocab_file)
            config_file = os.path.join(self.args.bert_init_dir, self.args.bert_config_file)
            des_config_file = os.path.join(self.args.model_save_dir, self.args.bert_config_file)

            # 如果是从原始BERT model开始保存的话，需要copy vocab和config过去
            if self.args.bert_init_dir != self.args.model_save_dir:
                shutil.copy(vocab_file, des_vocab_file)
                shutil.copy(config_file, des_config_file)

        model_file = os.path.join(self.args.model_save_dir, self.args.model_name + ".ckpt." + str(step))
        self.saver.save(session, model_file)
        return True

    def load_model(self, session, ckpt_step=""):
        """
        加载模型
        """
        suffix = ".ckpt.%s" % ckpt_step if ckpt_step != "" else ".ckpt"
        model_name = self.args.model_name + suffix
        all_init_checkpoint = os.path.join(self.args.model_save_dir, model_name)

        if not self.saver:
            self.saver = tf.train.Saver()
        self.saver.restore(session, all_init_checkpoint)
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

    if args.stage == "train":
        with tf.Session(config=config) as session:
            wrapper = BertWrapper(args)
            wrapper.init_data_socket()
            wrapper.init_data_source()
            wrapper.prepare_data_dict()
            wrapper.dump_environment()
            wrapper.build_graph(is_train=True)
            wrapper.train_procedure(session)
    elif args.stage == "continue_train":
        with tf.Session(config=config) as session:
            wrapper = BertWrapper(args)
            wrapper.init_data_socket()
            wrapper.load_environment(is_continue_train=True)
            wrapper.build_graph(is_train=True, is_continue_train=True)
            wrapper.load_model(session, ckpt_step=args.load_step)
            wrapper.train_procedure(session, is_continue_train=True)
    elif args.stage == "test":
        with tf.Session(config=config) as session:
            wrapper = BertWrapper(args)
            wrapper.init_data_socket()
            wrapper.load_environment()
            wrapper.build_graph(is_train=True)
            wrapper.load_model(session)
            wrapper.test_procedure(session, step=0)
    elif args.stage == "export":
        with tf.Session(config=config) as session:
            wrapper = BertWrapper(args)
            wrapper.load_environment()
            wrapper.build_graph(is_train=False)
            wrapper.export_model(session, args.signature_name, "ddl_graph")


if __name__ == "__main__":
    main()

