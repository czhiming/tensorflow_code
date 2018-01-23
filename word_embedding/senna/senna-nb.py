#encoding:utf-8

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

from input_data import *

import numpy as np
import tensorflow as tf
import time
import math

#参数集合
data_dir = '../data/xinhua'
batch_size = 120
win_size = 3
hidden_num = 256
word_dim = 100
neg_size = 10
num_epochs = 50
margin = 1.0
alpha = 0.05
grad_clip = 10.0

#准备训练数据
data_loader = TextLoader(data_dir, batch_size, win_size, neg_size)
vocab_size = data_loader.vocab_size

#准备测试例子
test_words = [u'贵州', u'今天', u'中国', u'一']

#模型定义
graph = tf.Graph()
with graph.as_default():
    #输入变量
    input_data = tf.placeholder(tf.int32, [batch_size, win_size * 2])
    targets_pos = tf.placeholder(tf.int64, [batch_size, 1])
    targets_neg = tf.placeholder(tf.int64, [batch_size, neg_size])

    #模型参数
    with tf.variable_scope('senna' + 'embedding'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, word_dim], -1.0, 1.0))
        embeddings = tf.nn.l2_normalize(embeddings, 1)

    with tf.variable_scope('senna' + 'weight'):
        weight_h = tf.Variable(tf.truncated_normal([win_size * 2 * word_dim + 1, hidden_num],
                        stddev=1.0 / math.sqrt(hidden_num)))
        softmax_w = tf.Variable(tf.truncated_normal([win_size * 2 * word_dim, word_dim],
                        stddev=1.0 / math.sqrt(word_dim)))
        softmax_u = tf.Variable(tf.truncated_normal([hidden_num + 1, word_dim],
                        stddev=1.0 / math.sqrt(hidden_num)))

    #得到上下文的隐藏层表示
    def infer_output(input_data):
        inputs_emb = tf.nn.embedding_lookup(embeddings, input_data)
        inputs_emb = tf.reshape(inputs_emb, [-1, win_size * 2 * word_dim])
        inputs_emb_add = tf.concat(1, [inputs_emb, tf.ones(tf.pack([tf.shape(input_data)[0], 1]))])

        inputs = tf.tanh(tf.matmul(inputs_emb_add, weight_h))
        inputs_add = tf.concat(1, [inputs, tf.ones(tf.pack([tf.shape(input_data)[0], 1]))])
        outputs = tf.matmul(inputs_add, softmax_u) + tf.matmul(inputs_emb, softmax_w)
        outputs = tf.clip_by_value(outputs, -10.0, 10.0)
        return outputs

    outputs = infer_output(input_data)
    pos_embs = tf.nn.embedding_lookup(embeddings, tf.squeeze(targets_pos))
    pos_scores = tf.reduce_sum(outputs * pos_embs, 1) #一个batch的正例得分

    loss, norm = 0.0, 0.0
    norm += tf.reduce_sum(tf.reduce_sum(tf.square(pos_embs), 1))
    for neg_targets in tf.split(1, neg_size, targets_neg):
        neg_embs = tf.nn.embedding_lookup(embeddings, tf.squeeze(neg_targets))
        neg_scores = tf.reduce_sum(outputs * neg_embs, 1) #一个batch的负例得分
        loss += tf.reduce_sum(tf.nn.relu(neg_scores + margin - pos_scores))
        norm += tf.reduce_sum(tf.reduce_sum(tf.square(neg_embs), 1))

    loss = loss + alpha * norm
    #self.optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    #输出词向量
    embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / embeddings_norm

#模型训练
with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    for e in range(num_epochs):
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            start = time.time()
            x, py, ny = data_loader.next_batch()
            feed = {input_data: x,
                    targets_pos: py,
                    targets_neg: ny}
            train_loss,  _ = sess.run([loss, optimizer], feed)
            end = time.time()
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                    b, data_loader.num_batches,
                    e, train_loss, end - start))
        np.save('word_embeddings', normalized_embeddings.eval())

#模型测试
final_embeddings = np.load('word_embeddings.npy')
for word in test_words:
    if not data_loader.vocab.has_key(word):
        continue
    word_vec = final_embeddings[data_loader.vocab.get(word),:]
    sim_mat = np.matmul(final_embeddings, word_vec)
    neareast = (-sim_mat).argsort()[1:11]
    neareast_words = [data_loader.words[id] for id in neareast]
    result_str = '与词<{0}>最相似的前10个词为：'.format(word) + ','.join(neareast_words)
    print result_str.encode('utf-8')
