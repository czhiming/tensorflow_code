#-*- coding:utf8 -*-
'''
Created on Jul 27, 2017

@author: czm
'''
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

result = a + b
print a.graph is tf.get_default_graph()












if __name__ == '__main__':
    pass
