#!/usr/bin/python
#_*_coding:utf-8_*_

import tensorflow as tf


def get_mask_from_tensor(tensor):
    """
    将向量转化为mask
    """
    x_mask = tf.sign(tf.abs(tf.reduce_sum(tensor, axis=-1)))
    return x_mask
