#-*- coding:utf8 -*-
'''
Created on 17-11-7

@author: czm
'''

import tensorflow as tf


tf.flags.DEFINE_string('filename', 'a.txt', 'a file name')


FLAGS = tf.flags.FLAGS

print FLAGS.filename

















if __name__ == '__main__':
    pass