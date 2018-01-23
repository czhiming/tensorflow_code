#-*- coding:utf8 -*-
'''
Created on Jul 4, 2017

@author: czm
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from click.core import batch


# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_steps = 3000
batch_size = 128

n_inputs = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # time steps
n_hidden_units = 128 # neurals in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define weights
weights = {
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])), #(28,128)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes])) #(128,10)
    }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])), #(128,)
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }

def RNN(X, weights, biases, batch_size=batch_size):
    # hidden layer for input to cell
    # ====================================
    # X -> (128, 28, 28)
    X = tf.reshape(X, [-1, n_inputs])
    
    X_in = tf.matmul(X, weights['in'] + biases['in'])
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    # basic LSTM Cell
    # ====================================
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    
    # hidden layer for output as the final results
    # ====================================
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    return results

def train(model):
    pred = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        
        while step < training_steps:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
            if step % 100 == 0:
                print sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}) 
                saver.save(sess,"rnn_model/"+model)

            step += 1
        
def test(model):
    
    # 加载测试数据
    X_test = mnist.test.images
    y_test = mnist.test.labels
    X_test = X_test.reshape(X_test.shape[0],n_steps, n_inputs)
    
    # 重新定义网络结构
    pred = RNN(x, weights, biases, batch_size=X_test.shape[0])
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # 加载训练好的模型，得出测试集的结果
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"rnn_model/"+model)
        print sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    
    
        

if __name__ == '__main__':
    #train('rnn_mnist.ckpt')
    test('rnn_mnist.ckpt')
    
    
    
    
    
    
    