#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import rnn

from lstm_bn import BatchNormLSTMCell


class Model():
    def __init__(self, args, deterministic=False):
        self.args = args

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'bn-lstm':
            cell_fn = BatchNormLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        if args.model == 'bn-lstm':
            cell = cell_fn(args.rnn_size, self.is_training)
        else:
            cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int64, [None, args.seq_length])
        # self.targets = tf.placeholder(tf.int64, [None, args.seq_length])  # seq2seq model
        self.targets = tf.placeholder(tf.int64, [None, ])  # target is class label

        with tf.variable_scope('embeddingLayer'):
            with tf.device('/cpu:0'):
                W = tf.get_variable('W', [args.vocab_size, args.rnn_size])
                embedded = tf.nn.embedding_lookup(W, self.input_data)

                # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
                inputs = tf.split(embedded, args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = rnn.static_rnn(self.cell, inputs, dtype=tf.float32, scope='rnnLayer')

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [args.rnn_size, args.label_size])
            softmax_b = tf.get_variable('b', [args.label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits)

        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.targets))  # Softmax loss
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.targets))  # Softmax loss
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)  # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def predict_label(self, sess, labels, text):
        x = np.array(text)
        feed = {self.input_data: x}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        id2labels = dict(zip(labels.values(), labels.keys()))
        labels = map(id2labels.get, results)
        return labels


    def predict_class(self, sess, text):
        x = np.array(text)
        feed = {self.input_data: x}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        return results
