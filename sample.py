#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import argparse
import cPickle as pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import TextLoader
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--how', type=str, default='sample',
                        help='sample or accuracy, test one sample or compute accuracy of dataset')
    parser.add_argument('--sample_text', type=str, default=' ',
                        help='sample text')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory containing input.csv')

    args = parser.parse_args()
    if args.how == 'sample':
        sample(args)
    elif args.how == 'accuracy':
        accuracy(args)
    else:
        raise Exception('incorrect argument, input "sample" or "accuracy" after "--how"')


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    with open(os.path.join(args.save_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)

    model = Model(saved_args, deterministic=True)

    text = map(vocab.get, args.sample_text.decode('utf8'))
    text = map(lambda i: i if i else 0, text)
    if len(text) >= saved_args.seq_length:
        text = text[:saved_args.seq_length]
    else:
        text = text + [0] * (saved_args.seq_length - len(text))

    with tf.Session() as sess:
        saver =tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print model.predict(sess, labels, [text])


def accuracy(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    with open(os.path.join(args.save_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)

    data_loader = TextLoader(args.data_dir, saved_args.batch_size, saved_args.seq_length)
    model = Model(saved_args, deterministic=True)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        correct_total = 0.0
        num_total = 0.0
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            start = time.time()
            state = model.initial_state.eval()
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y, model.initial_state: state}
            sub_accuracy, correct_num, probs = sess.run([model.accuracy, model.correct_num, model.probs], feed_dict=feed)
            end = time.time()
            # print '{}/{}, accuracy = {:.3f}, time/batch = {:.3f}'\
            #     .format(b+1,
            #             data_loader.num_batches,
            #             sub_accuracy,
            #             end - start)

            # ############
            # if b==0:
            #     d1 = dict(zip(vocab.values(), vocab.keys()))
            #     d2 = dict(zip(labels.values(), labels.keys()))
            #     for n, i in enumerate(x):
            #         s = []
            #         for j in i:
            #             if j:
            #                 s.append(d1[j])
            #         print ''.join(s), '\t', d2[y[n]], '\t', y[n], '\t', np.argmax(probs[n], 0)
            # ############

            correct_total += correct_num
            num_total += saved_args.batch_size

        accuracy_total = correct_total / num_total
        print 'total_num = {}, total_accuracy = {:.6f}'.format(int(num_total), accuracy_total)



if __name__ == '__main__':
    main()

