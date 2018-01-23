#encoding:utf-8

from input_data import *

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
import time


#参数集合
data_dir = '../data/tinyshakespeare'
batch_size = 120
seq_length = 10
hidden_num = 256
word_dim = 200
num_epochs = 50
model = 'lstm'
grad_clip = 10.0


#准备训练数据
data_loader = TextLoader2(data_dir, batch_size, seq_length)
vocab_size = data_loader.vocab_size
print 'vocab num: ', vocab_size

#模型定义
graph = tf.Graph()
with graph.as_default():

    if model == 'rnn':
        cell_fn = rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fn = rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fn = rnn_cell.BasicLSTMCell
    else:
        raise Exception("model type not supported: {}".format(model))

    cell = cell_fn(hidden_num)

    #输入变量
    input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    targets = tf.placeholder(tf.int64, [batch_size, seq_length])

    initial_state = cell.zero_state(batch_size, tf.float32)
    #模型参数
    with tf.variable_scope('rnnlm' + 'embedding'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, word_dim], -1.0, 1.0))
        embeddings = tf.nn.l2_normalize(embeddings, 1)

    with tf.variable_scope('rnnlm' + 'weight'):
        softmax_w = tf.get_variable("softmax_w", [hidden_num, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])

    # def loop(prev, _):
    #     prev = tf.matmul(prev, softmax_w) + softmax_b
    #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    #     return tf.nn.embedding_lookup(embeddings, prev_symbol)

    inputs = tf.split(1, seq_length,
                      tf.nn.embedding_lookup(embeddings, input_data))
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, last_state = seq2seq.rnn_decoder(inputs, initial_state, cell)
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_num])
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    loss_rnn = seq2seq.sequence_loss_by_example([logits],
            [tf.reshape(targets, [-1])],
            [tf.ones([batch_size * seq_length])],
            vocab_size)
    cost = tf.reduce_sum(loss_rnn) / batch_size / seq_length
    final_state = last_state
    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
            grad_clip)
    optimizer = tf.train.AdagradOptimizer(0.1)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

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
            x, y = data_loader.next_batch()
            feed = {input_data: x, targets: y}
            train_loss,  _ = sess.run([cost, train_op], feed)
            end = time.time()
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                    b, data_loader.num_batches,
                    e, train_loss, end - start))
        np.save('rnnlm_word_embeddings', normalized_embeddings.eval())