#-*- coding:utf8 -*-
'''
Created on 17-10-17

@author: czm
'''
import tensorflow as tf
from utils import pp


def ff_layer_init(options,params,prefix='ff_',nin=None,nout=None):
    if nin == None and nout == None:
        return 0
    with tf.variable_scope(prefix):
        params[pp(prefix,'W')] = tf.Variable(tf.truncated_normal([nin, nout]))
        params[pp(prefix,'b')] = tf.Variable(tf.truncated_normal([nout,]))
    return params

def ff_layer(params,emb,options,prefix='ff_',activ=tf.nn.tanh):
    return activ(tf.matmul(emb,params[pp(prefix,'W')])+\
                 params[pp(prefix,'b')])



if __name__ == '__main__':
    pass