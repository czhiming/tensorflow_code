#-*- coding:utf8 -*-
'''
Created on Jul 1, 2017
@function: MNIST 手写数字识别
@author: czm
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers.core import Activation
from click.core import batch

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b) 
    
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
    
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784]) # 28x28
ys = tf.placeholder(tf.float32,[None,10])

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1])) # loss

#optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5)
optimizer = tf.train.GradientDescentOptimizer(0.3)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        # print sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}),
        print compute_accuracy(mnist.test.images, mnist.test.labels)
        










if __name__ == '__main__':
    pass