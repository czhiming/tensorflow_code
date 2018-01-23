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
batch_size = 1200
win_size = 3
word_dim = 100
num_epochs = 100
alpha = 0.05
cooccurrence_cap = 100
scaling_factor = 0.75

#准备训练数据
data_loader = TextLoader(data_dir, batch_size, win_size)
vocab_size = data_loader.vocab_size
#准备测试例子
test_words = [u'贵州', u'今天', u'中国', u'一']

#模型定义
graph = tf.Graph()
with graph.as_default():
    #输入变量
    focal_input = tf.placeholder(tf.int32, shape=[batch_size])
    context_input = tf.placeholder(tf.int32, shape=[batch_size])
    cooccurrence_count = tf.placeholder(tf.float32, shape=[batch_size])

    #模型参数
    with tf.variable_scope('glove' + 'embedding'):
        focal_embeddings = tf.Variable(
            tf.random_uniform([vocab_size, word_dim], 1.0, -1.0))
        context_embeddings = tf.Variable(
            tf.random_uniform([vocab_size, word_dim], 1.0, -1.0))
        focal_embeddings = tf.nn.l2_normalize(focal_embeddings, 1)
        context_embeddings = tf.nn.l2_normalize(context_embeddings, 1)

        focal_biases = tf.Variable(tf.random_uniform([vocab_size], 1.0, -1.0))
        context_biases = tf.Variable(tf.random_uniform([vocab_size], 1.0, -1.0))

    #计算过程
    focal_embs = tf.nn.embedding_lookup([focal_embeddings], focal_input)
    context_embs = tf.nn.embedding_lookup([context_embeddings], context_input)
    focal_bias = tf.nn.embedding_lookup([focal_biases], focal_input)
    context_bias = tf.nn.embedding_lookup([context_biases], context_input)

    weighting_factor = tf.minimum(1.0, tf.pow(
            tf.div(cooccurrence_count, cooccurrence_cap), scaling_factor))

    embedding_product = tf.reduce_sum(tf.mul(focal_embs, context_embs), 1)

    log_cooccurrences = tf.log(tf.to_float(cooccurrence_count))

    distance_expr = tf.square(tf.add_n([
        embedding_product,
        focal_bias,
        context_bias,
        tf.neg(log_cooccurrences)]))

    single_losses = tf.mul(weighting_factor, distance_expr)
    loss = tf.reduce_sum(single_losses)

    optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)

    #输出词向量
    combined_embeddings = tf.add(focal_embeddings, context_embeddings)
    embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(combined_embeddings), 1, keep_dims=True))
    normalized_embeddings = combined_embeddings / embeddings_norm


#模型训练
with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    #for e in range(num_epochs):
    for e in range(2):
        data_loader.reset_batch_pointer()
        start = time.time()
        loss_all = 0
        #for b in range(data_loader.num_batches):
        for b in range(3):
            rows, cols, vals = data_loader.next_batch()
            feed = {focal_input: rows,
                    context_input: cols,
                    cooccurrence_count: vals}
            loss_val,  _ = sess.run([loss, optimizer], feed)
            loss_all += loss_val
        end = time.time()
        print("{}/{}, train_loss = {:.3f}, time/batch = {:.3f}" .format(
            e, num_epochs, loss_all / data_loader.num_batches, end - start))
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
    print 'the nearest word with {0} is:'.format(word.encode('utf-8')) + ','.join(neareast_words).encode('utf-8')
