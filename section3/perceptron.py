#-*- coding:utf8 -*-
'''
Created on Jun 30, 2017

@author: czm
'''
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input,in_size,out_size,activation_func=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input,Weights) + biases
    
    if activation_func is None:
        return Wx_plus_b
    else:
        return activation_func(Wx_plus_b)


x_data = np.linspace(-1,1,300)[:,np.newaxis].astype('float32')
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

#一定需要定义类型
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
#定义结构
layer1 = add_layer(xs,1,10,activation_func=tf.nn.relu)
prediction = add_layer(layer1,10,1,activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data-prediction),
                     reduction_indices=[1]))
#少了minimize
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion() #不暂停继续往下走
plt.show()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lines = None
    for i in range(5000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 100 == 0:
#             print i,
#             print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.2)






if __name__ == '__main__':
    pass