#-*- coding:utf8 -*-
'''
Created on Jun 30, 2017

@author: czm
'''

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input,in_size,out_size,n_layer,activation_func=None):
    with tf.name_scope('layer'):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.contrib.deprecated.histogram_summary(layer_name+'/Weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
            tf.contrib.deprecated.histogram_summary(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input,Weights) + biases
        
        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs =  activation_func(Wx_plus_b)
        tf.contrib.deprecated.histogram_summary(layer_name+'/outputs',outputs)
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis].astype('float32')
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

#一定需要定义类型
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
#定义结构
layer1 = add_layer(xs,1,10,n_layer=1,activation_func=tf.nn.relu)
prediction = add_layer(layer1,10,1,n_layer=2,activation_func=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data-prediction),
                     reduction_indices=[1]))
    tf.contrib.deprecated.scalar_summary('loss',loss)
#少了minimize
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
merged = tf.contrib.deprecated.merge_all_summaries()
writer = tf.summary.FileWriter("logs",sess.graph)

sess.run(tf.global_variables_initializer())
    
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)







if __name__ == '__main__':
    pass