#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Customized LSTM Cell based on Basic on LSTMCell of tensorflow.
Add batch-normalization layer before activation.

Refer to github: https://github.com/ScartleRoy/TF_LSTM_seq_bn
Refer to paper: https://arxiv.org/pdf/1603.09025v4.pdf
'''


import collections
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell import _get_concat_variable
from tensorflow.python.ops.rnn_cell import _get_sharded_variable

from tensorflow.python.ops.nn import batch_normalization, moments
from tensorflow.python.training.moving_averages import ExponentialMovingAverage

from tensorflow.python.ops.control_flow_ops import cond



def batch_norm(x, deterministic, alpha=0.9, shift=True, scope='bn'):
    with vs.variable_scope(scope):
        dtype = x.dtype
        input_shape = x.get_shape().as_list()
        feat_dim = input_shape[-1]
        axes = range(len(input_shape)-1)

        if shift:
            beta = vs.get_variable(
                scope+"_beta", shape=[feat_dim],
                initializer=init_ops.zeros_initializer, dtype=dtype)
        else:
            beta = vs.get_variable(
                scope+"_beta", shape=[feat_dim],
                initializer=init_ops.zeros_initializer,
                dtype=dtype, trainable=False)

        gamma = vs.get_variable(
            scope+"_gamma", shape=[feat_dim],
            initializer=init_ops.constant_initializer(0.1), dtype=dtype)

        mean = vs.get_variable(scope+"_mean", shape=[feat_dim],
                               initializer=init_ops.zeros_initializer,
                               dtype=dtype, trainable=False)

        var = vs.get_variable(scope+"_var", shape=[feat_dim],
                              initializer=init_ops.ones_initializer,
                              dtype=dtype, trainable=False)

        counter = vs.get_variable(scope+"_counter", shape=[],
                                  initializer=init_ops.constant_initializer(0),
                                  dtype=tf.int64, trainable=False)

        zero_cnt = vs.get_variable(scope+"_zero_cnt", shape=[],
                                   initializer=init_ops.constant_initializer(0),
                                   dtype=tf.int64, trainable=False)

        batch_mean, batch_var = moments(x, axes, name=scope+'_moments')

        mean, var = cond(math_ops.equal(counter, zero_cnt), lambda: (batch_mean, batch_var),
                         lambda: (mean, var))

        mean, var, counter = cond(deterministic, lambda: (mean, var, counter),
                                  lambda: ((1-alpha) * batch_mean + alpha * mean,
                                           (1-alpha) * batch_var + alpha * var,
                                           counter + 1))
        normed = batch_normalization(x, mean, var, beta, gamma, 1e-8)

    return normed



class BNLSTMCell(RNNCell):

    def __init__(self,
                 num_units,
                 input_size=None,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 num_unit_shards=1,
                 num_proj_shards=1,
                 forget_bias=1.0,
                 bn=0,
                 return_gate=False,
                 deterministic=None):

        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell
          input_size: int, The dimensionality of the inputs into the LSTM cell
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          num_unit_shards: How to split the weight matrix.  If >1, the weight
            matrix is stored across num_unit_shards.
          num_proj_shards: How to split the projection matrix.  If >1, the
            projection matrix is stored across num_proj_shards.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of the training.
          bn: int, set 1,2 or 3 to enable sequence-wise batch normalization with
            different level. Implemented according to arXiv:1603.09025
          return_gate: bool, set true to return the values of the gates.
          deterministic: Tensor, control training and testing phase, decide whether to
            open batch normalization.
        """
        self._num_units = num_units
        self._input_size = input_size
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._bn = bn
        self._return_gate = return_gate
        self._deterministic = deterministic

        if num_proj:
            self._state_size = num_units + num_proj
            self._output_size = num_proj
        else:
            self._state_size = 2 * num_units
            self._output_size = num_units


    @property
    def input_size(self):
        return self._num_units if self._input_size is None else self._input_size


    @property
    def output_size(self):
        return self._output_size


    @property
    def state_size(self):
        return self._state_size


    def __call__(self, inputs, state, scope=None):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: state Tensor, 2D, batch x state_size.
          scope: VariableScope for the created subgraph; defaults to "LSTMCell".

        Returns:
          A tuple containing:
          - A 2D, batch x output_dim, Tensor representing the output of the LSTM
            after reading "inputs" when previous state was "state".
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - A 2D, batch x state_size, Tensor representing the new state of LSTM
            after reading "inputs" when previous state was "state".
        Raises:
          ValueError: if an input_size was specified and the provided inputs have
            a different dimension.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
        m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        actual_input_size = inputs.get_shape().as_list()[1]
        if self._input_size and self._input_size != actual_input_size:
            raise ValueError("Actual input size not same as specified: %d vs %d." %
                             (actual_input_size, self._input_size))

        scope_name = scope or type(self).__name__
        with vs.variable_scope(scope_name,
                               initializer=self._initializer):  # "LSTMCell"
            if not self._bn:
                concat_w = _get_concat_variable(
                    "W", [actual_input_size + num_proj, 4 * self._num_units],
                    dtype, self._num_unit_shards)
            else:
                concat_w_i = _get_concat_variable(
                    "W_i", [actual_input_size, 4 * self._num_units],
                    dtype, self._num_unit_shards)
                concat_w_r = _get_concat_variable(
                    "W_r", [num_proj, 4 * self._num_units],
                    dtype, self._num_unit_shards)

            b = vs.get_variable(
                "B", shape=[4 * self._num_units],
                initializer=array_ops.zeros_initializer, dtype=dtype)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            if not self._bn:
                cell_inputs = array_ops.concat(1, [inputs, m_prev])
                lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)

            else:
                lstm_matrix_i = batch_norm(math_ops.matmul(inputs, concat_w_i), self._deterministic,
                                           shift=False, scope=scope_name+'bn_i')
                if self._bn > 1:
                    lstm_matrix_r = batch_norm(math_ops.matmul(m_prev, concat_w_r), self._deterministic,
                                               shift=False, scope=scope_name+'bn_r')
                else:
                    lstm_matrix_r = math_ops.matmul(m_prev, concat_w_r)
                lstm_matrix = nn_ops.bias_add(math_ops.add(lstm_matrix_i, lstm_matrix_r), b)

            i, j, f, o = array_ops.split(1, 4, lstm_matrix)

            # Diagonal connections
            if self._use_peepholes:
                w_f_diag = vs.get_variable(
                    "W_F_diag", shape=[self._num_units], dtype=dtype)
                w_i_diag = vs.get_variable(
                    "W_I_diag", shape=[self._num_units], dtype=dtype)
                w_o_diag = vs.get_variable(
                    "W_O_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                     sigmoid(i + w_i_diag * c_prev) * tanh(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * tanh(j))

            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            if self._use_peepholes:
                if self._bn > 2:
                    m = sigmoid(o + w_o_diag * c) * tanh(batch_norm(c, self._deterministic,
                                                                    scope=scope_name+'bn_m'))
                else:
                    m = sigmoid(o + w_o_diag * c) *tanh(c)
            else:
                if self._bn > 2:
                    m = sigmoid(o) * tanh(batch_norm(c, self._deterministic,
                                                     scope=scope_name+'bn_m'))
                else:
                    m = sigmoid(o) * tanh(c)

            if self._num_proj is not None:
                concat_w_proj = _get_concat_variable(
                    "W_P", [self._num_units, self._num_proj],
                    dtype, self._num_proj_shards)

                m = math_ops.matmul(m, concat_w_proj)

        if not self._return_gate:
            return m, array_ops.concat(1, [c, m])
        else:
            return m, array_ops.concat(1, [c, m]), (i, j, f, o)
