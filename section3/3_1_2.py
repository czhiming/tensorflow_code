#-*- coding:utf8 -*-
'''
Created on Jun 26, 2017

@author: czm
'''
from __future__ import print_function
import tensorflow as tf

def program1():
    a = tf.constant([1.,2.],name='a')
    b = tf.constant([2.,3.],name='b')
    result = a + b
    
    print(a.graph is tf.get_default_graph()) #获取默认的计算图

def program2():
    g1 = tf.Graph()
    with g1.as_default():
        v = tf.get_variable("v",initializer=tf.zeros_initializer(shape=[1]))

def program3():
    import numpy as np
    array = np.random.rand(32, 100, 100)

    def my_func(arg):
      arg = tf.convert_to_tensor(arg, dtype=tf.float32)
      return tf.matmul(arg, arg) + arg
    
    # The following calls are equivalent.
    value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
    value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
    value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    with tf.Session() as sess:
        print(sess.run(value_2))

def program4():
    a = tf.zeros([1,2])
    print(a)






if __name__ == '__main__':
    program4()
    
    