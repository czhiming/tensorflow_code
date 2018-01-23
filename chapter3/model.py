#-*- coding:utf8 -*-
'''
Created on Jul 28, 2017

@author: czm
'''
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1+v2

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, "model/model.ckpt")

#直接获取计算图
saver = tf.train.import_meta_graph("model/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")
    print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))






if __name__ == '__main__':
    pass