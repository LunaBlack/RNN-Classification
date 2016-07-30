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

    parser.add_argument('--utils_dir', type=str, default='data',
                        help='''directory containing labels.pkl and corpus.txt:
                        'corpus.txt'      : corpus to define vocabulary;
                        'vocab.pkl'       : vocabulary definitions;
                        'labels.pkl'      : label definitions''')

    parser.add_argument('--data_path', type=str, default='data/input.csv',
                        help='data to train model')

    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')

    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm or bn-lstm, default lstm')

    parser.add_argument('--bn_level', type=int, default=1,
                        help='if model is bn-lstm, enable sequence-wise batch normalization with different level')

    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in RNN')

    parser.add_argument('--batch_size', type=int, default=32,
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

    parser.add_argument('--init_from', type=str, default=None,
                        help='''continue training from saved model at this path. Path must contain files saved by previous training process:
                        'config.pkl'         : configuration;
                        'checkpoint'         : paths to model file(s) (created by tensorflow).
                                               Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'       : file(s) with model definition (created by tensorflow)''')

    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextLoader(args.utils_dir, args.data_path, args.batch_size, args.seq_length, None, None)
    args.vocab_size = data_loader.vocab_size
    args.label_size = data_loader.label_size

    if args.init_from is not None:
        assert os.path.isdir(args.init_from), '%s must be a path' % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, 'config.pkl')), 'config.pkl file does not exist in path %s' % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, 'chars_vocab.pkl')), 'chars_vocab.pkl file does not exist in path %s' % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, 'labels.pkl')), 'labels.pkl file does not exist in path %s' % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pickle.load(f)
        need_be_same = ['model', 'rnn_size', 'num_layers', 'seq_length']
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme], 'command line argument and saved model disagree on %s' % checkme

        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = pickle.load(f)
        with open(os.path.join(args.init_from, 'labels.pkl'), 'rb') as f:
            saved_labels = pickle.load(f)
        assert saved_chars==data_loader.chars, 'data and loaded model disagree on character set'
        assert saved_vocab==data_loader.vocab, 'data and loaded model disagree on dictionary mappings'
        assert saved_labels==data_loader.labels, 'data and loaded model disagree on label dictionary mappings'

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.chars, data_loader.vocab), f)
    with open(os.path.join(args.save_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(data_loader.labels, f)

    model = Model(args)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.all_variables())

        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()

            for b in range(data_loader.num_batches):
                start = time.time()
                state = model.initial_state.eval()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
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



if __name__ == '__main__':
    main()

