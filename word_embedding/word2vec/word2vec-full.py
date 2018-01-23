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
    parser.add_argument('--batch_size', type=int, default=120,
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
    data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size, mini_frq=50)
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
            embeddings = tf.nn.l2_normalize(embeddings, 1)

            nce_weights = tf.Variable(tf.truncated_normal([args.vocab_size, args.word_dim],
                                                          stddev=1.0 / math.sqrt(args.word_dim)))
            nce_biases = tf.Variable(tf.zeros([args.vocab_size]))

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        labels = tf.expand_dims(train_labels, 1)

        if labels.dtype != tf.int64:
            labels = tf.cast(labels, tf.int64)
        labels_flat = tf.reshape(labels, [-1])

        #第一部分,抽取负例,计算正负例得分
        sampled, true_expected_count, sampled_expected_count = tf.nn.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=1,
          num_sampled=args.neg_size,
          unique=True,
          range_max=args.vocab_size)

        all_ids = tf.concat(0, [labels_flat, sampled])

        all_w = tf.nn.embedding_lookup(nce_weights, all_ids) #[batch+neg,dim]
        all_b = tf.nn.embedding_lookup(nce_biases, all_ids) #[batch+neg]

        true_w = tf.slice(all_w, tf.pack([0, 0]), [args.batch_size, args.word_dim])
        true_b = tf.slice(all_b, [0], [args.batch_size])
        true_logits = tf.matmul(embed, true_w, transpose_b=True) + true_b

        sampled_w = tf.slice(all_w, tf.pack([args.batch_size, 0]), [args.neg_size, args.word_dim])
        sampled_b = tf.slice(all_b, [args.batch_size], [args.neg_size])
        sampled_logits = tf.matmul(embed, sampled_w, transpose_b=True) + sampled_b

        if True: #减去词出现的先验频率
          true_logits -= tf.log(true_expected_count)
          sampled_logits -= tf.log(sampled_expected_count)
        out_logits = tf.concat(1, [true_logits, sampled_logits])
        out_targets = tf.concat(1, [tf.ones_like(true_logits), tf.zeros_like(sampled_logits)])

        #第二部分：计算正负例与正确标签的交叉熵交叉熵
        #logits,[batch,1+neg_num],[batch,1+neg_num]
        #sigmoid_cross_entropy_with_logits(logits, targets)
        loss_batchs = tf.nn.relu(out_logits) - out_logits * out_targets + tf.log(1 + tf.exp(-tf.abs(out_logits)))
        loss = tf.reduce_mean(tf.reduce_sum(loss_batchs, 1))

        # loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed,
        #                                      labels, args.neg_size, args.vocab_size))

        optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        #输出词向量
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm


    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
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
                print 'loss each batch:',loss_val
            end = time.time()
            print("{}/{}, train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        e, args.num_epochs, loss_all/data_loader.num_batches, end - start))
                # print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                #         b, data_loader.num_batches,
                #         e, train_loss, end - start))
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