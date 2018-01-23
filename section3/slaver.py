#-*- coding:utf8 -*-
'''
Created on Jul 4, 2017
@function: 保存和提取
@author: czm
'''

import tensorflow as tf
import numpy as np

# save to file
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
# 
# init = tf.global_variables_initializer()
# 
# saver = tf.train.Saver()
# 
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"my_net/save_net.ckpt")
#     print "Save to path:", save_path

# restore variables
# 只保存了数据，没保存框架
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,
                name="weights") # name 不能漏掉
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,
                name="biases")

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print "Weights:", sess.run(W)
    print "biases:", sess.run(b)


if __name__ == '__main__':
    pass