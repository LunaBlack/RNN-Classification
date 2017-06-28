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
    rootdir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--utils_dir', type=str, default=rootdir+'/utils/',
                        help='''directory containing labels.pkl and corpus.txt:
                        'corpus.txt'      : corpus to define vocabulary;
                        'vocab.pkl'       : vocabulary definitions;
                        'labels.pkl'      : label definitions''')

    parser.add_argument('--data_path', type=str, default=rootdir+'/data/data.csv',
                        help='data to train model')

    parser.add_argument('--save_dir', type=str, default=rootdir+'/save/',
                        help='directory to store checkpointed models')

    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm or bn-lstm, default lstm')

    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in RNN')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')

    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    parser.add_argument('--save_every', type=int, default=100,
                        help='save frequency')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate for rmsprop')

    args = parser.parse_args()
    cross_validation(args)


def cross_validation(args):
    data_loader = TextLoader(True, args.utils_dir, args.data_path, args.batch_size, args.seq_length, None, None)
    args.vocab_size = data_loader.vocab_size
    args.label_size = data_loader.label_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.utils_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.chars, data_loader.vocab), f)
    with open(os.path.join(args.utils_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(data_loader.labels, f)

    data = data_loader.tensor.copy()
    np.random.shuffle(data)
    data_list = np.array_split(data, 10, axis=0)

    model = Model(args)
    accuracy_list = []

    with tf.Session() as sess:
        for n in range(10):
            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver(tf.all_variables())

            test_data = data_list[n].copy()
            train_data = np.concatenate(map(lambda i: data_list[i], [j for j in range(10) if j!=n]), axis=0)
            data_loader.tensor = train_data

            for e in range(args.num_epochs):
                sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
                data_loader.reset_batch_pointer()

                for b in range(data_loader.num_batches):
                    start = time.time()
                    x, y = data_loader.next_batch()
                    feed = {model.input_data: x, model.targets: y}
                    train_loss, state, _, accuracy = sess.run([model.cost, model.final_state, model.optimizer, model.accuracy], feed_dict=feed)
                    end = time.time()
                    print '{}/{} (epoch {}), train_loss = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}'\
                        .format(e * data_loader.num_batches + b + 1,
                                args.num_epochs * data_loader.num_batches,
                                e + 1,
                                train_loss,
                                accuracy,
                                end - start)
                    if (e*data_loader.num_batches+b+1) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b==data_loader.num_batches-1):
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=e*data_loader.num_batches+b+1)
                        print 'model saved to {}'.format(checkpoint_path)

            n_chunks = len(test_data) / args.batch_size
            if len(test_data) % args.batch_size:
                n_chunks += 1
            test_data_list = np.array_split(test_data, n_chunks, axis=0)

            correct_total = 0.0
            num_total = 0.0
            for m in range(n_chunks):
                start = time.time()
                x = test_data_list[m][:, :-1]
                y = test_data_list[m][:, -1]
                results = model.predict_class(sess, x)
                correct_num = np.sum(results==y)
                end = time.time()

                correct_total += correct_num
                num_total += len(x)

            accuracy_total = correct_total / num_total
            accuracy_list.append(accuracy_total)
            print 'total_num = {}, total_accuracy = {:.6f}'.format(int(num_total), accuracy_total)

    accuracy_average = np.average(accuracy_list)
    print 'The average accuracy of cross_validation is {}'.format(accuracy_average)



if __name__ == '__main__':
    main()
