#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.models.rnn import rnn, rnn_cell

from lstm_bn import BNLSTMCell


class Model():
    def __init__(self, args, deterministic=False):
        self.args = args

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'bn-lstm':
            cell_fn = BNLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        deterministic = tf.Variable(deterministic, name='deterministic')  # when training, set to False; when testing, set to True
        if args.model == 'bn-lstm':
            cell = cell_fn(args.rnn_size, bn=args.bn_level, deterministic=deterministic)
        else:
            cell = cell_fn(args.rnn_size)
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int64, [None, args.seq_length])
        # self.targets = tf.placeholder(tf.int64, [None, args.seq_length])  # seq2seq model
        self.targets = tf.placeholder(tf.int64, [None, ])  # target is class label
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('embeddingLayer'):
            with tf.device('/cpu:0'):
                W = tf.get_variable('W', [args.vocab_size, args.rnn_size])
                embedded = tf.nn.embedding_lookup(W, self.input_data)

                # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
                inputs = tf.split(1, args.seq_length, embedded)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = rnn.rnn(cell, inputs, self.initial_state, scope='rnnLayer')

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [args.rnn_size, args.label_size])
            softmax_b = tf.get_variable('b', [args.label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits)

        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.targets))  # Softmax loss
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.targets))  # Softmax loss
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)  # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def predict_label(self, sess, labels, text):
        x = np.array(text)
        state = self.cell.zero_state(len(text), tf.float32).eval()
        feed = {self.input_data: x, self.initial_state: state}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        id2labels = dict(zip(labels.values(), labels.keys()))
        labels = map(id2labels.get, results)
        return labels


    def predict_class(self, sess, text):
        x = np.array(text)
        state = self.cell.zero_state(len(text), tf.float32).eval()
        feed = {self.input_data: x, self.initial_state: state}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        return results
