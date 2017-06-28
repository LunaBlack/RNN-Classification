#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import commands
import subprocess


def run(rootdir):
    command1 = 'cd {rootdir}'.format(rootdir=rootdir)

    # 训练模型
    command2 = '''python {train_py} \
                    --utils_dir         {utils_dir} \
                    --data_path         {data_path} \
                    --save_dir          {save_dir} \
                    --model             {model} \
                    --rnn_size          {rnn_size} \
                    --num_layers        {num_layers} \
                    --batch_size        {batch_size} \
                    --seq_length        {seq_length} \
                    --num_epochs        {num_epochs} \
                    --save_every        {save_every} \
                    --learning_rate     {learning_rate} \
                    --decay_rate        {decay_rate} \
                    --continue_training {continue_training}'''.format(train_py         ='train.py',
                                                                      utils_dir        ='utils',
                                                                      data_path        ='data/train.csv',
                                                                      save_dir         ='save',
                                                                      model            ='lstm',  # rnn/gru/lstm/bn-lstm
                                                                      rnn_size         =128,
                                                                      num_layers       =1,
                                                                      batch_size       =128,
                                                                      seq_length       =20,
                                                                      num_epochs       =100,
                                                                      save_every       =100,
                                                                      learning_rate    =0.001,
                                                                      decay_rate       =0.9,
                                                                      continue_training='False')

    # 测试模型
    command3 = '''python {test_py} \
                    --save_dir          {save_dir} \
                    --how               {how} \
                    --sample_text       {sample_text} \
                    --data_path         {data_path} \
                    --result_path       {result_path}'''.format(test_py         ='test.py',
                                                                save_dir        ='save',
                                                                how             ='accuracy',  # sample为测试单个例子，sample_text不能为None；predict为预测多个例子；accuracy为预测并检验多个例子
                                                                sample_text     =' ',
                                                                data_path       ='data/test.csv',  # predict和accuracy模式下必需
                                                                result_path     ='data/result.csv')  # predict模式下必需

    # 交叉验证
    command4 = '''python {cross_py} \
                    --utils_dir         {utils_dir} \
                    --data_path         {data_path} \
                    --save_dir          {save_dir} \
                    --model             {model} \
                    --rnn_size          {rnn_size} \
                    --num_layers        {num_layers} \
                    --batch_size        {batch_size} \
                    --seq_length        {seq_length} \
                    --num_epochs        {num_epochs} \
                    --save_every        {save_every} \
                    --learning_rate     {learning_rate} \
                    --decay_rate        {decay_rate}'''.format(cross_py         ='cross_validation.py',
                                                               utils_dir        ='utils',
                                                               data_path        ='data/data.csv',
                                                               save_dir         ='save',
                                                               model            ='lstm',  # rnn/gru/lstm/bn-lstm
                                                               rnn_size         =128,
                                                               num_layers       =1,
                                                               batch_size       =128,
                                                               seq_length       =10,
                                                               num_epochs       =100,
                                                               save_every       =100,
                                                               learning_rate    =0.001,
                                                               decay_rate       =0.9)

    subprocess.call(command1, shell=True)

    t1 = time.time()
    subprocess.call(command2, shell=True)
    t2 = time.time()
    print 'training costs time: ', t2-t1

    t1 = time.time()
    subprocess.call(command3, shell=True)
    t2 = time.time()
    print 'testing costs time: ', t2 - t1

    t1 = time.time()
    subprocess.call(command4, shell=True)
    t2 = time.time()
    print 'cross validation costs time: ', t2-t1



if __name__ == '__main__':

    rootdir = os.path.dirname(__file__)
    run(rootdir)
