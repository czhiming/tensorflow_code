#-*- coding:utf8 -*-
'''
Created on Jul 27, 2017

@author: czm
'''
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer(shape=[1]))

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run() #初始化所有变量
    with tf.variable_scope("",reuse=True): #重新使用
        print sess.run(tf.get_variable("v"))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print sess.run(tf.get_variable("v"))





if __name__ == '__main__':
    pass