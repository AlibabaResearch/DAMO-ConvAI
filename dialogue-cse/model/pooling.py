#!/usr/bin/python
#_*_coding:utf-8_*_

import tensorflow as tf

from model import operations


def attentive_pooling(inputs, attention_size, sequence_mask=None, return_alphas=False, scope="", temperature=1.0):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    """
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = inputs[0] + inputs[1]

    inputs_shape = inputs.shape # ( batch_size , seq_len, hidden_size)
    sequence_length = inputs_shape[1].value  # the length of sequences processed
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as _:
        w = tf.get_variable("w", shape=[hidden_size, attention_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable("b", shape=[attention_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        u = tf.get_variable("u", shape=[attention_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w) + tf.reshape(b, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u, [-1, 1])) / temperature

    vu = tf.reshape(vu, [-1, sequence_length])
    if sequence_mask is not None:
        zero_pad = tf.ones_like(vu) * (-2 ** 32 + 1)
        vu = tf.where(sequence_mask > 0.0, vu, zero_pad)  # 不必担心浮点数比较问题
        score = tf.nn.softmax(vu)
    else:
        score = tf.nn.softmax(vu)
    output = tf.reduce_sum(inputs * tf.reshape(score, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    return output, score


def average_pooling(inputs, sequence_mask=None):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    """
    if sequence_mask is None:
        sequence_mask = operations.get_mask_from_tensor(inputs)

    inputs = tf.reduce_sum(inputs * tf.expand_dims(sequence_mask, dim=2), axis=1) / (
        tf.expand_dims(tf.reduce_sum(sequence_mask, axis=1), dim=1) + 1e-3)
    return inputs
