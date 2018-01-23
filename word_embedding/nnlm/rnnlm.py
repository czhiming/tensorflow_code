#encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from input_data import *

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
import argparse
import time
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/xinhua',
                       help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=120,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=5,
                       help='RNN sequence length')
    parser.add_argument('--hidden_num', type=int, default=256,
                       help='number of hidden layers')
    parser.add_argument('--word_dim', type=int, default=256,
                       help='number of word embedding')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--grad_clip', type=float, default=10.,
                       help='clip gradients at this value')

    args = parser.parse_args() #参数集合

    #准备训练数据
    data_loader = TextLoader2(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    #模型定义
    graph = tf.Graph()
    with graph.as_default():

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.hidden_num)

        #输入变量
        input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        targets = tf.placeholder(tf.int64, [args.batch_size, args.seq_length])

        initial_state = cell.zero_state(args.batch_size, tf.float32)
        #模型参数
        with tf.variable_scope('rnnlm' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))
            embeddings = tf.nn.l2_normalize(embeddings, 1)

        with tf.variable_scope('rnnlm' + 'weight'):
            softmax_w = tf.get_variable("softmax_w", [args.hidden_num, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        # def loop(prev, _):
        #     prev = tf.matmul(prev, softmax_w) + softmax_b
        #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        #     return tf.nn.embedding_lookup(embeddings, prev_symbol)

        inputs = tf.split(1, args.seq_length,
                          tf.nn.embedding_lookup(embeddings, input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = seq2seq.rnn_decoder(inputs, initial_state, cell)
        output = tf.reshape(tf.concat(1, outputs), [-1, args.hidden_num])
        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
        loss_rnn = seq2seq.sequence_loss_by_example([logits],
                [tf.reshape(targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        cost = tf.reduce_sum(loss_rnn) / args.batch_size / args.seq_length
        final_state = last_state
        lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdagradOptimizer(0.1)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        #输出词向量
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm

    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {input_data: x, targets: y}
                train_loss,  _ = sess.run([cost, train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))
            np.save('rnnlm_word_embeddings', normalized_embeddings.eval())

if __name__ == '__main__':
    main()