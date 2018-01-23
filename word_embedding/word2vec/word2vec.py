#encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from input_data import *

import numpy as np
import tensorflow as tf
import argparse
import time
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/text8',
                       help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=1200,
                       help='minibatch size')
    parser.add_argument('--win_size', type=int, default=3,
                       help='RNN sequence length')
    parser.add_argument('--word_dim', type=int, default=256,
                       help='number of word embedding')
    parser.add_argument('--neg_size', type=int, default=64,
                       help='number of negative words')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')

    args = parser.parse_args() #参数集合

    #准备训练数据
    data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size)
    args.vocab_size = data_loader.vocab_size
    #准备测试例子
    #test_words = [u'贵州', u'今天', u'中国', u'一']
    test_words = ['China', 'good', 'new', 'one']

    #模型定义
    graph = tf.Graph()
    with graph.as_default():
        #输入变量
        train_inputs = tf.placeholder(tf.int32, shape=[args.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[args.batch_size])

        #模型参数
        with tf.variable_scope('word2vec' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim],
                                                       -1.0, 1.0))
            embeddings = tf.nn.l2_normalize(embeddings, 1) # l2正则化

            nce_weights = tf.Variable(tf.truncated_normal([args.vocab_size, args.word_dim],
                                                          stddev=1.0 / math.sqrt(args.word_dim)))
            nce_biases = tf.Variable(tf.zeros([args.vocab_size]))

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        labels = tf.expand_dims(train_labels, 1)

        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels,
                                             embed, args.neg_size, args.vocab_size)) # tf.nn.nce_loss 损失函数

        optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        #输出词向量
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) # 保持张量的维数不变，即不会出现(2,)
        normalized_embeddings = embeddings / embeddings_norm


    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            start = time.time()
            loss_all = 0
            for b in range(data_loader.num_batches):
                batch_inputs, batch_labels = data_loader.next_batch()
                feed = {train_inputs: batch_inputs,
                        train_labels: batch_labels}
                loss_val,  _ = sess.run([loss, optimizer], feed)
                loss_all += loss_val
            end = time.time()
            print("{}/{}, train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        e, args.num_epochs, loss_all/data_loader.num_batches, end - start))
            np.save('word_embeddings', normalized_embeddings.eval())

            final_embeddings = np.load('word_embeddings.npy')
            for word in test_words:
                if not data_loader.vocab.has_key(word):
                    continue
                word_vec = final_embeddings[data_loader.vocab.get(word),:]
                sim_mat = np.matmul(final_embeddings, word_vec)
                neareast = (-sim_mat).argsort()[:11]
                neareast_words = [data_loader.words[id] for id in neareast]
                print '与词<{0}>最相似的前10个词为：'.format(word) + ','.join(neareast_words)

    #模型测试    final_embeddings = np.load('word_embeddings.npy')
    for word in test_words:
        if not data_loader.vocab.has_key(word):
            continue
        word_vec = final_embeddings[data_loader.vocab.get(word),:]
        sim_mat = np.matmul(final_embeddings, word_vec)
        neareast = (-sim_mat).argsort()[1:11]
        neareast_words = [data_loader.words[id] for id in neareast]
        print '与词<{0}>最相似的前10个词为：'.format(word) + ','.join(neareast_words)

if __name__ == '__main__':
    main()