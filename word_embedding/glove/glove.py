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
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=1200,
                       help='minibatch size')
    parser.add_argument('--win_size', type=int, default=3,
                       help='window sequence length')
    parser.add_argument('--word_dim', type=int, default=100,
                       help='number of word embedding')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='the weight of normalizer')
    parser.add_argument('--cooccurrence_cap', type=int, default=100)
    parser.add_argument('--scaling_factor', type=float, default=0.75)

    args = parser.parse_args() #参数集合

    #准备训练数据
    data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size)
    args.vocab_size = data_loader.vocab_size
    #准备测试例子
    test_words = ['good', 'today', 'china', 'one']

    #模型定义
    graph = tf.Graph()
    with graph.as_default():
        #输入变量
        focal_input = tf.placeholder(tf.int32, shape=[args.batch_size])
        context_input = tf.placeholder(tf.int32, shape=[args.batch_size])
        cooccurrence_count = tf.placeholder(tf.float32, shape=[args.batch_size])

        #模型参数
        with tf.variable_scope('glove' + 'embedding'):
            focal_embeddings = tf.Variable(
                tf.random_uniform([args.vocab_size, args.word_dim], 1.0, -1.0))
            context_embeddings = tf.Variable(
                tf.random_uniform([args.vocab_size, args.word_dim], 1.0, -1.0))
            focal_embeddings = tf.nn.l2_normalize(focal_embeddings, 1)
            #l2归一化
            context_embeddings = tf.nn.l2_normalize(context_embeddings, 1)

            focal_biases = tf.Variable(tf.random_uniform([args.vocab_size], 1.0, -1.0)) # (1024,1)
            context_biases = tf.Variable(tf.random_uniform([args.vocab_size], 1.0, -1.0)) # (1024,1)

        #计算过程
        focal_embs = tf.nn.embedding_lookup([focal_embeddings], focal_input) # (1024,100)
        context_embs = tf.nn.embedding_lookup([context_embeddings], context_input) # (1024, 100)
        focal_bias = tf.nn.embedding_lookup([focal_biases], focal_input)
        context_bias = tf.nn.embedding_lookup([context_biases], context_input)
        
        #cooccurrence_cap > x_max   求f(x)的值
        weighting_factor = tf.minimum(1.0, tf.pow(
                tf.div(cooccurrence_count, args.cooccurrence_cap), args.scaling_factor))
        
        embedding_product = tf.reduce_sum(tf.multiply(focal_embs, context_embs), 1) # (1024,)
        log_cooccurrences = tf.log(tf.to_float(cooccurrence_count)) # (1024,)
        # tf.add_n 把一个列表的东西依次加起来
        distance_expr = tf.square(tf.add_n([
            embedding_product,
            focal_bias,
            context_bias,
            tf.negative(log_cooccurrences)]))
        
        single_losses = tf.multiply(weighting_factor, distance_expr)
        loss = tf.reduce_sum(single_losses)

        optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)

        #输出词向量
        combined_embeddings = tf.add(focal_embeddings, context_embeddings)
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(combined_embeddings), 1, keep_dims=True))
        normalized_embeddings = combined_embeddings / embeddings_norm


    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            start = time.time()
            loss_all = 0
            for b in range(data_loader.num_batches):
                rows, cols, vals = data_loader.next_batch()
                feed = {focal_input: rows,
                        context_input: cols,
                        cooccurrence_count: vals}
                loss_val,  _ = sess.run([loss, optimizer], feed)
                
                loss_all += loss_val
            end = time.time()
            print("{}/{}, train_loss = {:.3f}, time/batch = {:.3f}" .format(
                e, args.num_epochs, loss_all / data_loader.num_batches, end - start))
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
        print '与词<{0}>最相似的前10个词为：'.format(word) + ','.join(neareast_words)

if __name__ == '__main__':
    main()
    
    
    
    