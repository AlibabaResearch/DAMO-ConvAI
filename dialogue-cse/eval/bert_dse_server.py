#!/usr/bin/python
# _*_coding:utf-8_*_

import os
import sys

import numpy as np
import tensorflow as tf

import wrapper_bert as wrapper
from config_bert import get_parser


class BertRepresentationServer(object):
    """
    基于QA关系抽取句子向量
    """
    def __init__(self, logger=None):
        # 模型和dictionary的版本对应需靠人工确保
        self.logger = logger
        self.graph = None
        self.sess = None
        self.wrapper = None

    def load_model(self, model_dir, env_name, ckpt_step):
        """
        加载某业务线的模型; use_char,seq_len为校验参数
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.wrapper = wrapper.BertWrapper(args=get_parser())
            self.wrapper.load_environment(model_dir, env_name)
            self.wrapper.build_graph(is_train=False)
            self.wrapper.load_model(self.sess, ckpt_step=ckpt_step)

    def encoding(self, batch_texts, is_batch=False, is_response=False):
        """
        评估批量query和answer的匹配情况，batch_query和batch_answer为原始文本列表的形式
        """
        def_representation = None

        if not is_batch:
            batch_texts = [batch_texts]

        # 不允许传入numpy形式的参数, 因为不保证一些操作适用于numpy
        if not isinstance(batch_texts, list):
            err_msg = u"native inference error: batch dimension type not list"
            self.logger.error(err_msg) if self.logger else print(err_msg, file=sys.stderr)
            return def_representation

        data_provider = self.wrapper.data_provider
        batch_data = data_provider.process_for_inference(batch_texts, batch_texts)  # 都用context占位

        batch_size = len(batch_texts)
        gpu_num = 1
        round_num = data_provider.round_num
        max_seq_len = data_provider.max_seq_len
        response_num = data_provider.test_response_num

        context_shape = [batch_size, gpu_num, round_num, max_seq_len]
        context_input_ids = np.reshape(batch_data.context_input_ids, context_shape)
        context_input_masks = np.reshape(batch_data.context_input_masks, context_shape)
        context_segment_ids = np.reshape(batch_data.context_segment_ids, context_shape)
        context_rounds = np.reshape(batch_data.context_rounds, [batch_size, gpu_num])

        next_context_input_ids = np.reshape(batch_data.next_context_input_ids, context_shape)
        next_context_input_masks = np.reshape(batch_data.next_context_input_masks, context_shape)
        next_context_segment_ids = np.reshape(batch_data.next_context_segment_ids, context_shape)
        next_context_rounds = np.reshape(batch_data.next_context_rounds, [batch_size, gpu_num])

        response_shape = [batch_size, gpu_num, response_num, max_seq_len]
        response_input_ids = np.reshape(batch_data.response_input_ids, response_shape)
        response_input_masks = np.reshape(batch_data.response_input_masks, response_shape)
        response_segment_ids = np.reshape(batch_data.response_segment_ids, response_shape)

        label_shape = [batch_size, gpu_num, response_num]
        labels = np.reshape(batch_data.labels, label_shape)

        feed_dict = {
            self.wrapper.context_input_ids: context_input_ids,
            self.wrapper.context_input_masks: context_input_masks,
            self.wrapper.context_segment_ids: context_segment_ids,
            self.wrapper.rounds: context_rounds,
            self.wrapper.next_context_input_ids: next_context_input_ids,
            self.wrapper.next_context_input_masks: next_context_input_masks,
            self.wrapper.next_context_segment_ids: next_context_segment_ids,
            self.wrapper.next_rounds: next_context_rounds,
            self.wrapper.response_input_ids: response_input_ids,
            self.wrapper.response_input_masks: response_input_masks,
            self.wrapper.response_segment_ids: response_segment_ids,
            self.wrapper.labels: labels
        }

        # 结果产出shape是[gpu_num, batch_size, feature] or [gpu_num, batch_size, response_num, feature]
        try:
            feature = self.wrapper.test_response_feature if is_response is True else self.wrapper.test_context_feature
            representations = self.sess.run(feature, feed_dict=feed_dict)
        except Exception as e:
            err_msg = "native rank error: %s" % e
            self.logger.error(err_msg) if self.logger else print(err_msg, file=sys.stderr)
            return def_representation

        # 取第0个GPU, context为[1, batch, 1, feature], response为[1, batch, response_num, feature]
        representations = representations[0]
        if not is_batch:
            representations = representations[0]
        return representations

    def unload_model(self):
        """
        卸载当前model
        """
        self.sess.close()
