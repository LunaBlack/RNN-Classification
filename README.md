# RNN-Classification  
本项目构建了基于RNN/LSTM模型的文本分类器，基于TensorFlow r1.0版本。

This project constructs a text classifier based on RNN/LSTM model, based on TensorFlow r1.0.

当前支持BASIC-RNN、GRU、LSTM、BN-LSTM。

This project currently supports the BASIC-RNN, GRU, LSTM, and BN-LSTM models.

## 运行方式
 + 命令行，运行train.py训练模型，运行test.py测试模型，所需参数用help查看。
 + 运行run.py文件，自行修改各项参数

## 代码说明
+ model.py: 根据给定的参数、用tensorflow构建分类模型
+ lstm_bn.py: 根据论文[*Recurrent Batch Normalization*](http://arxiv.org/abs/1603.09025)构建Batch Normalization的LSTM的Cell单元
+ train.py: 模型的训练
+ test.py: 模型的测试，分为sample（预测单个例子）、predict（预测文件中的所有例子）、accuracy（预测文件中的所有例子并验证）
+ cross_validation.py: 十折交叉验证
+ utils.py: 辅助函数，主要用于得到训练/测试的批数据（batch data）
+ run.py: 训练、测试、交叉验证模型，用python运行命令行，可自行修改各项参数
+ data: 训练数据、测试数据、十折交叉验证数据，第一列为文本text，第二列为类label，可修改位置并修改run.py文件或命令行运行的参数
+ save: 保存模型训练的各项参数，可修改位置并修改run.py文件或命令行运行的参数
+ utils: 语料、label对照字典等，可修改位置并修改run.py文件或命令行运行的参数
 - corpus.txt: 语料，*需要提前生成*
 - labels.pkl: 类标号与数字的对照，字典形式{类标号:从0开始的数字}，pickle形式，*需要提前生成*
