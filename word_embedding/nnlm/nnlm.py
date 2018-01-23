#encoding:utf-8

import sys
from input_data import *

import numpy as np
import tensorflow as tf
import time
import math

#######################
# 设置的参数
#######################
data_dir = 'data/text8'
vocab_dir = 'data/vocab.json'
batch_size = 120
win_size = 5
hidden_num = 256
word_dim = 100
num_epochs = 10
grad_clip = 10.0


#准备训练数据
data_loader = TextLoader(data_dir, vocab_dir, batch_size, win_size)
vocab_size = data_loader.vocab_size


#准备测试例子
test_words_str = ['I','come','from', 'north','america']
test_words_ids = [data_loader.vocab.get(w, 1) for w in test_words_str]

#模型定义
graph = tf.Graph()
with graph.as_default():
    #输入变量
    input_data = tf.placeholder(tf.int32, [batch_size, win_size]) # X
    targets = tf.placeholder(tf.int64, [batch_size, 1]) # y
    test_words = tf.placeholder(tf.int64, [win_size])

    #模型参数
    with tf.variable_scope('nnlm' + 'embedding'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, word_dim], -1.0, 1.0))
        embeddings = tf.nn.l2_normalize(embeddings, 1) # L2 范数归一化

    with tf.variable_scope('nnlm' + 'weight'):
        weight_h = tf.Variable(tf.truncated_normal([win_size * word_dim + 1, hidden_num],
                        stddev=1.0 / math.sqrt(hidden_num)))
        softmax_w = tf.Variable(tf.truncated_normal([win_size * word_dim, vocab_size],
                        stddev=1.0 / math.sqrt(win_size * word_dim)))
        softmax_u = tf.Variable(tf.truncated_normal([hidden_num + 1, vocab_size],
                        stddev=1.0 / math.sqrt(hidden_num)))


    #得到上下文的隐藏层表示
    def infer_output(input_data):
        inputs_emb = tf.nn.embedding_lookup(embeddings, input_data)
        inputs_emb = tf.reshape(inputs_emb, [-1, win_size * word_dim]) # 注意此处的操作
        inputs_emb_add = tf.concat([inputs_emb, tf.ones((tf.shape(input_data)[0], 1))], 1) # 增加一列 1

        inputs = tf.tanh(tf.matmul(inputs_emb_add, weight_h))  # tanh(Hx+d)
        inputs_add = tf.concat([inputs, tf.ones((tf.shape(input_data)[0], 1))], 1)
        outputs = tf.matmul(inputs_add, softmax_u) + tf.matmul(inputs_emb, softmax_w) # y = Utanh(Hx+d)+WX+b
        outputs = tf.clip_by_value(outputs, 0.0, grad_clip) # 限制值在某个范围
        outputs = tf.nn.softmax(outputs) # softmax函数求得概率
        return outputs

    outputs = infer_output(input_data)
    one_hot_targets = tf.one_hot(tf.squeeze(targets), vocab_size, 1.0, 0.0) # squeeze 挤压矩阵 （100,1） -> (100,)

    loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1)) # 对数似然损失函数
    optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

    #输出词向量
    test_outputs = infer_output(tf.expand_dims(test_words_ids, 0)) # 扩充tensor的维数
    test_outputs = tf.arg_max(test_outputs, 1)

    embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / embeddings_norm

#模型训练
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for e in range(num_epochs):
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            start = time.time()
            x, y = data_loader.next_batch()
            feed = {input_data: x, targets: y}
            train_loss,  _ = sess.run([loss, optimizer], feed)
            end = time.time()
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                    b, data_loader.num_batches,
                    e, train_loss, end - start))
        np.save('nnlm_word_embeddings', normalized_embeddings.eval())

    #样例测试
    feed = {test_words : test_words_ids}
    [test_outputs] = sess.run([test_outputs], feed)
    print ' '.join(test_words_str)
    print data_loader.words[test_outputs[0]]
    
    

