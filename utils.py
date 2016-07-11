#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import csv
import collections
import cPickle as pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, 'input.csv')
        label_file = os.path.join(data_dir, 'labels.pkl')
        vocab_file = os.path.join(data_dir, 'vocab.pkl')
        tensor_file = os.path.join(data_dir, 'data.npy')

        with open(label_file, 'r') as f:
            self.labels = pickle.load(f)
        self.label_size = len(self.labels)

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print 'reading text file'
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print 'loading preprocessed files'
            self.load_preprocessed(vocab_file, tensor_file)

        self.reset_batch_pointer()


    def transform(self, d):
        if len(d) >= self.seq_length:
            new_d = map(self.vocab.get, d[:self.seq_length])
        else:
            new_d = map(self.vocab.get, d) + [0] * (self.seq_length - len(d))
        return new_d


    def preprocess(self, input_file, vocab_file, tensor_file):
        data = pd.read_csv(input_file, encoding='utf8')

        counter = collections.Counter(''.join(data['text'].values))
        count_pairs = sorted(counter.items(), key=lambda i: -i[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars) + 1
        self.vocab = dict(zip(self.chars, range(1, len(self.chars)+1)))

        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)

        tensor_x = np.array(list(map(self.transform, data['text'])))
        tensor_y = np.array(list(map(self.labels.get, data['label'])))
        self.tensor = np.c_[tensor_x, tensor_y].astype(int)
        np.save(tensor_file, self.tensor)


    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        self.vocab_size = len(self.chars) + 1
        self.vocab = dict(zip(self.chars, range(1, len(self.chars)+1)))
        self.tensor = np.load(tensor_file)


    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'

        np.random.shuffle(self.tensor)
        tensor = self.tensor[:self.num_batches * self.batch_size]
        self.x_batches = np.split(tensor[:, :-1], self.num_batches, 0)
        self.y_batches = np.split(tensor[:, -1], self.num_batches, 0)


    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        self.pointer += 1
        return x, y


    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0

