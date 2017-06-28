#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn.python.ops import core_rnn_cell



class BatchNormLSTMCell(core_rnn_cell.RNNCell):
    """
    Batch normalized LSTM as described in http://arxiv.org/abs/1603.09025
    """

    def __init__(self, num_units, is_training=False, forget_bias=1.0, activation=tanh, reuse=None):
        self._num_units = num_units
        self._is_training = is_training
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse


    @property
    def state_size(self):
        return core_rnn_cell.LSTMStateTuple(self._num_units, self._num_units)


    @property
    def output_size(self):
        return self._num_units


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self._reuse):
            c, h = state
            input_size = inputs.get_shape().as_list()[1]

            W_xh = tf.get_variable('W_xh',
                                   [input_size, 4 * self._num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                                   [self._num_units, 4 * self._num_units],
                                   initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self._num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm(xh, 'xh', self._is_training)
            bn_hh = batch_norm(hh, 'hh', self._is_training)

            hidden = bn_xh + bn_hh + bias

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=hidden, num_or_size_splits=4, axis=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            bn_new_c = batch_norm(new_c, 'c', self._is_training)
            new_h = self._activation(bn_new_c) * sigmoid(o)
            new_state = core_rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state



def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def bn_lstm_identity_initializer(scale):

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        """
        Ugly cause LSTM params calculated in one matrix multiply
        """

        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype=dtype)

    return _initializer


def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer


def batch_norm(x, name_scope, is_training, epsilon=1e-3, decay=0.999):
    """
    Assume 2d [batch, values] tensor
    """

    with tf.variable_scope(name_scope):
        training = tf.constant(is_training)
        size = x.get_shape().as_list()[1]

        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer(), trainable=False)

        def batch_statistics():
            batch_mean, batch_var = tf.nn.moments(x, [0])
            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)
