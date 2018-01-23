#-*- coding:utf8 -*-
'''
Created on Jun 30, 2017

@author: czm
'''
import tensorflow as tf 

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1,matrix2)

#method1
# sess = tf.Session()
# result = sess.run(product)
# 
# print result
# sess.close()

#method2

with tf.Session() as sess:
    print sess.run(product)





if __name__ == '__main__':
    pass