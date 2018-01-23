#-*- coding:utf8 -*-
'''
Created on Jul 2, 2017

@author: czm
'''
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import sys

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y) # 二值化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print X_train[:10]
print X_test.shape[:10]


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(biases)
        print sess.run(Weights).shape
        print inputs
        #sys.exit(0)
        
    
    # here to dropout
    #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs',outputs)
    
# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64]) # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'layer1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'layer2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                               reduction_indices=[1])) # loss

tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""
sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


print sess.run(l1, feed_dict={xs: X_train})


for i in range(500):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train}) # 修改此处
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train}) 
        test_result = sess.run(merged, feed_dit={xs: X_test, ys: y_test})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

sess.close()
"""

    
if __name__ == '__main__':
    pass