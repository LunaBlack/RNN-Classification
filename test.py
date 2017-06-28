#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import csv
import argparse
import cPickle as pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import TextLoader
from model import Model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--utils_dir', type=str, default='utils',
                        help='''directory containing labels.pkl and corpus.txt:
                        'corpus.txt'      : corpus to define vocabulary;
                        'vocab.pkl'       : vocabulary definitions;
                        'labels.pkl'      : label definitions''')

    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')

    parser.add_argument('--how', type=str, default='accuracy',
                        help='''sample / predict / accuracy:
                        test one sample / predict some samples / compute accuracy of dataset''')

    parser.add_argument('--sample_text', type=str, default=' ',
                        help='sample text, necessary when how is sample')

    parser.add_argument('--data_path', type=str, default='data/test.csv',
                        help='data to predict or compute accuracy, necessary when how is predict or accuracy')

    parser.add_argument('--result_path', type=str, default='data/result.csv',
                        help='result of prediction, necessary when how is predict')

    args = parser.parse_args()
    if args.how == 'sample':
        sample(args)
    elif args.how == 'predict':
        predict(args)
    elif args.how == 'accuracy':
        accuracy(args)
    else:
        raise Exception('incorrect argument, input "sample" or "accuracy" after "--how"')


def transform(text, seq_length, vocab):
    x = map(vocab.get, text)
    x = map(lambda i: i if i else 0, x)
    if len(x) >= seq_length:
        x = x[:seq_length]
    else:
        x = x + [0] * (seq_length - len(x))
    return x


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)

    model = Model(saved_args, deterministic=True)
    x = transform(args.sample_text.decode('utf8'), saved_args.seq_length, vocab)

    with tf.Session() as sess:
        saver =tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print model.predict_label(sess, labels, [x])


def predict(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)

    model = Model(saved_args, deterministic=True)

    with open(args.data_path, 'r') as f:
        reader = csv.reader(f)
        texts = list(reader)

    texts = map(lambda i: i[0], texts)
    x = map(lambda i: transform(i.strip().decode('utf8'), saved_args.seq_length, vocab), texts)

    with tf.Session() as sess:
        saver =tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        start = time.time()
        results = model.predict_label(sess, labels, x)
        end = time.time()
        print 'prediction costs time: ', end - start

    with open(args.result_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(texts, results))


def accuracy(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    with open(os.path.join(args.utils_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)

    data_loader = TextLoader(False, args.utils_dir, args.data_path, saved_args.batch_size, saved_args.seq_length, vocab, labels)
    model = Model(saved_args, deterministic=True)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        data = data_loader.tensor.copy()
        n_chunks = len(data) / saved_args.batch_size
        if len(data) % saved_args.batch_size:
            n_chunks += 1
        data_list = np.array_split(data, n_chunks, axis=0)

        correct_total = 0.0
        num_total = 0.0
        for m in range(n_chunks):
            start = time.time()
            x = data_list[m][:, :-1]
            y = data_list[m][:, -1]
            results = model.predict_class(sess, x)
            correct_num = np.sum(results==y)
            end = time.time()
            print 'batch {}/{} cost time {:.3f}, sub_accuracy = {:.6f}'.format(m+1, n_chunks, end-start, correct_num*1.0/len(x))

            correct_total += correct_num
            num_total += len(x)

        accuracy_total = correct_total / num_total
        print 'total_num = {}, total_accuracy = {:.6f}'.format(int(num_total), accuracy_total)



if __name__ == '__main__':
    main()

