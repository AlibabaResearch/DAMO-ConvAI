#!/usr/bin/python
# _*_coding:utf-8_*_

import tensorflow as tf
from tensorflow.python.ops import array_ops

from model import pooling

from model import bert
from util.bert import run_classifier


class CLBERT(object):
    """
    建立CLBERTT模型
    """

    def __init__(self, model_params):
        self.model_params = model_params
        self.round_num = model_params["round_num"]
        self.max_seq_len = model_params["max_seq_len"]
        self.hidden_size = model_params["hidden_size"]
        self.attention_size = 256
        self.class_num = model_params["class_num"]
        self.bert_config = model_params["bert_config"]
        self.bert_hidden_size = self.bert_config.hidden_size
        self.init_checkpoint = model_params["init_checkpoint"]
        self.bidirectional = model_params["bidirectional"]
        self.layer_num = 1  # model_params["layer_num"]

        # initializer
        self.random_initializer = tf.random_uniform_initializer(-1.0, 1.0)
        self.xavier_initializer = tf.contrib.layers.xavier_initializer()
        self.ortho_initializer = tf.orthogonal_initializer()

    def get_cos_distance(self, X1, X2):
        # calculate cos distance between two sets
        # more similar more big
        X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=2))
        X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=2))

        X1_X2 = tf.reduce_sum(X1 * X2, axis=2)
        X1_X2_norm = X1_norm * X2_norm

        cos = X1_X2 / (X1_X2_norm)
        cos = cos / 0.1
        return cos

    def matching_feature(self, context_embeds, context_masks, resp_embeds):
        """
        计算匹配特征
        """
        # norm_val = tf.sqrt(tf.to_float(self.bert_hidden_size))
        norm_val = 1.0
        matching_feature_list = []

        for i in range(context_embeds.shape.as_list()[1]):
            matching_feature = tf.matmul(
                tf.tile(context_embeds[:, i:i + 1, :, :], [1, array_ops.shape(resp_embeds)[1], 1, 1]),
                tf.transpose(resp_embeds, perm=[0, 1, 3, 2])) / norm_val
            matching_feature = matching_feature * 0.1

            matching_feature_list.append(tf.matmul(matching_feature, resp_embeds))
        matching_feature = tf.stack(matching_feature_list, axis=2)
        matching_feature = tf.reduce_sum(matching_feature, axis=3) / tf.cast(
            tf.reduce_sum(tf.expand_dims(context_masks, dim=1), axis=3, keep_dims=True), dtype=tf.float32)
        return matching_feature

    def forward(self, context_input_ids, context_input_masks, context_segment_ids, rounds, next_context_input_ids,
            next_context_input_masks, next_context_segment_ids, next_rounds, response_input_ids, response_input_masks,
            response_segment_ids, roles, data_types, answer_num, is_training, use_init_model=False):
        """
        在每个对应设备上建立单个模型图
        """
        init_checkpoint = self.init_checkpoint if use_init_model is True else None
        batch_size = array_ops.shape(context_input_ids)[0]

        # context encoder
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as sc:
            context_shape = [batch_size * self.round_num, self.max_seq_len]
            context_input_ids = tf.reshape(context_input_ids, context_shape)
            context_input_masks = tf.reshape(context_input_masks, context_shape)
            context_segment_ids = tf.reshape(context_segment_ids, context_shape)

            context_outputs = run_classifier.create_model(self.bert_config, is_training, context_input_ids,
                context_input_masks, context_segment_ids, sc.name, init_checkpoint, use_seq_feature=True)
            context_outputs = context_outputs[-self.layer_num:]

            # 先进行mask消除不存在位置的embedding
            token_mask = tf.expand_dims(tf.cast(context_input_masks, dtype=tf.float32), dim=2)
            context_outputs = [t * token_mask for t in context_outputs]  # 从底向上
            # reshape原始output为round形式
            outputs_shape = [-1, self.round_num, self.max_seq_len, self.bert_hidden_size]
            context_outputs = [tf.reshape(t, outputs_shape) for t in context_outputs]

            # token层面聚合
            context_sentence_embeddings = []
            for i, layer_context_outputs in enumerate(context_outputs):
                layer_sentence_embeddings = tf.reduce_sum(layer_context_outputs, axis=2) / (tf.reduce_sum(
                    tf.reshape(token_mask, [-1, self.round_num, self.max_seq_len, 1]), axis=2) + 1e-6)
                context_sentence_embeddings.append(layer_sentence_embeddings)

            round_index = rounds - 1
            gather_index = tf.stack([tf.range(0, batch_size), round_index], axis=1)
            return_context_embedding = tf.add_n(context_sentence_embeddings[-self.layer_num:])
            return_context_embedding = tf.gather_nd(return_context_embedding, gather_index)
            return_context_embedding = tf.expand_dims(return_context_embedding, axis=1)

        # next context encoder
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as sc:
            next_context_shape = [batch_size * self.round_num, self.max_seq_len]
            next_context_input_ids = tf.reshape(next_context_input_ids, next_context_shape)
            next_context_input_masks = tf.reshape(next_context_input_masks, next_context_shape)
            next_context_segment_ids = tf.reshape(next_context_segment_ids, next_context_shape)

            # 因为和上文通用，此处不使用init_checkpoint
            next_context_outputs = run_classifier.create_model(self.bert_config, is_training, next_context_input_ids,
                next_context_input_masks, next_context_segment_ids, sc.name, None, use_seq_feature=True)
            next_context_outputs = next_context_outputs[-self.layer_num:]

            # 先进行mask消除不存在位置的embedding
            token_mask = tf.expand_dims(tf.cast(next_context_input_masks, dtype=tf.float32), dim=2)
            next_context_outputs = [t * token_mask for t in next_context_outputs]  # 从底向上
            # reshape原始output为round形式
            outputs_shape = [-1, self.round_num, self.max_seq_len, self.bert_hidden_size]
            next_context_outputs = [tf.reshape(t, outputs_shape) for t in next_context_outputs]

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as sc:
            response_shape = [batch_size * answer_num, self.max_seq_len]
            response_input_ids = tf.reshape(response_input_ids, response_shape)
            response_input_masks = tf.reshape(response_input_masks, response_shape)
            response_segment_ids = tf.reshape(response_segment_ids, response_shape)

            response_outputs = run_classifier.create_model(self.bert_config, is_training, response_input_ids,
                response_input_masks, response_segment_ids, sc.name, None, use_seq_feature=True)
            response_outputs = response_outputs[-self.layer_num:]

            # 先进行mask消除不存在位置的embedding
            token_mask = tf.expand_dims(tf.cast(response_input_masks, dtype=tf.float32), dim=2)
            response_outputs = [t * token_mask for t in response_outputs]  # 从底向上
            # reshape原始output为round形式
            outputs_shape = [-1, answer_num, self.max_seq_len, self.bert_hidden_size]
            response_outputs = [tf.reshape(t, outputs_shape) for t in response_outputs]

            # token层面聚合
            response_sentence_embeddings = []
            for i, layer_response_outputs in enumerate(response_outputs):
                layer_sentence_mebeddings = tf.reduce_sum(layer_response_outputs, axis=2) / (tf.reduce_sum(
                    tf.reshape(token_mask, [-1, answer_num, self.max_seq_len, 1]), axis=2) + 1e-6)
                response_sentence_embeddings.append(layer_sentence_mebeddings)

            return_response_embedding = tf.add_n(response_sentence_embeddings[-self.layer_num:])[:, 0:1, :]

        fw_round_mask = tf.sequence_mask(rounds, self.round_num, dtype=tf.float32)
        bw_round_mask = tf.sequence_mask(next_rounds, self.round_num, dtype=tf.float32)

        context_input_masks = tf.reshape(context_input_masks, [-1, self.round_num, self.max_seq_len])
        next_context_input_masks = tf.reshape(next_context_input_masks, [-1, self.round_num, self.max_seq_len])
        response_input_masks = tf.reshape(response_input_masks, [-1, answer_num, self.max_seq_len])

        last_context_output = context_outputs[-1]
        last_next_context_output = next_context_outputs[-1]
        last_response_output = response_outputs[-1]

        with tf.variable_scope("round_attention", reuse=tf.AUTO_REUSE):
            context_round_embedding = tf.reduce_mean(last_context_output, axis=2)
            next_context_round_embedding = tf.reduce_mean(last_next_context_output, axis=2)

            _, context_score = pooling.attentive_pooling(context_round_embedding, self.attention_size,
                sequence_mask=fw_round_mask, return_alphas=True)
            _, last_context_score = pooling.attentive_pooling(next_context_round_embedding, self.attention_size,
                sequence_mask=bw_round_mask, return_alphas=True)

            context_score = tf.tile(tf.reshape(context_score, [-1, 1, self.round_num, 1]), [1, answer_num, 1, 1])
            last_context_score = tf.tile(tf.reshape(last_context_score, [-1, 1, self.round_num, 1]), [1, answer_num, 1, 1])

        fw_mge = self.matching_feature(last_context_output, context_input_masks, last_response_output)
        bw_mge = self.matching_feature(last_next_context_output, next_context_input_masks, last_response_output)

        fw_mge = fw_mge * (1.0 + context_score)
        bw_mge = bw_mge * (1.0 + last_context_score)

        if self.bidirectional == 1:
            fw_mge = tf.reduce_sum(fw_mge + bw_mge, axis=2)
        else:
            fw_mge = tf.reduce_sum(fw_mge, axis=2)

        estimation = self.get_cos_distance(response_sentence_embeddings[-1], fw_mge)
        tf_operation = {"estimation": estimation, "score": estimation, "attention_score": 1,
            "context_feature": return_context_embedding, "response_feature": return_response_embedding}
        return tf_operation

    def calculate_loss(self, multi_estimations, labels):
        """
        logits 是未经过sigmoid的值, 目前本模型仅支持multi模式
        """
        loss = -tf.reduce_mean(tf.nn.log_softmax(multi_estimations, axis=-1) * labels)
        return loss


def main():
    test_params = {"attention_probs_dropout_prob": 0.1, "directionality": "bidi", "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1, "hidden_size": 768, "initializer_range": 0.02, "intermediate_size": 3072,
        "max_position_embeddings": 512, "num_attention_heads": 12, "num_hidden_layers": 12, "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12, "pooler_num_fc_layers": 3, "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform", "type_vocab_size": 2, "vocab_size": 21128}

    model_params = {"multi_mode": True, "max_seq_len": 12, "layer_num": 2, "round_num": 3, "hidden_size": 768,
        "bidirectional": 1, "bert_config": bert.BertConfig.from_dict(test_params),
        # bert.BertConfig.from_json_file(/data/bert_chinese/bert_config.json"),
        "init_checkpoint": "/data/bert_chinese/bert_model.ckpt", "class_num": 2}

    answer_num = 10

    # input data
    context_input_ids = tf.placeholder(tf.int32, [256, model_params["round_num"], model_params["max_seq_len"]])
    context_input_masks = tf.placeholder(tf.int32, [256, model_params["round_num"], model_params["max_seq_len"]])
    context_segment_ids = tf.placeholder(tf.int32, [256, model_params["round_num"], model_params["max_seq_len"]])
    rounds = tf.placeholder(tf.int32, [256], name="round")

    next_context_input_ids = tf.placeholder(tf.int32, [256, model_params["round_num"], model_params["max_seq_len"]])
    next_context_input_masks = tf.placeholder(tf.int32, [256, model_params["round_num"], model_params["max_seq_len"]])
    next_context_segment_ids = tf.placeholder(tf.int32, [256, model_params["round_num"], model_params["max_seq_len"]])
    next_rounds = tf.placeholder(tf.int32, [256], name="next_round")

    response_input_ids = tf.placeholder(tf.int32, [256, answer_num, model_params["max_seq_len"]])
    response_input_masks = tf.placeholder(tf.int32, [256, answer_num, model_params["max_seq_len"]])
    response_segment_ids = tf.placeholder(tf.int32, [256, answer_num, model_params["max_seq_len"]])

    roles = tf.placeholder(tf.int32, [256, model_params["round_num"] * 2 + 1], name="roles")
    data_types = tf.placeholder(tf.int32, [256], name="data_types")
    label = tf.placeholder(tf.float32, [256, answer_num], name="label")

    model = CLBERT(model_params)
    out = model.forward(context_input_ids, context_input_masks, context_segment_ids, rounds, next_context_input_ids,
        next_context_input_masks, next_context_segment_ids, next_rounds, response_input_ids, response_input_masks,
        response_segment_ids, roles, data_types, answer_num, is_training=True, use_init_model=False)

    if model_params["multi_mode"] is False:
        loss = model.calculate_loss(out["estimation"], label)
        print(loss)
    else:
        multi_loss = model.calculate_loss(out["estimation"], label)
        print(multi_loss)


if __name__ == "__main__":
    main()


