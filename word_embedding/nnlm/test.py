#-*- coding:utf8 -*-
'''
Created on 17-10-17

@author: czm
'''
import json
from input_data import  *
from utils import load_config,load_dict
from nnlm import Graph
import tensorflow as tf
import numpy as np
import sys

def prob(
    lines,
    save_to='model/nnlm.ckpt',
    vocab='data/vocab.json',
    iter=300 #第几次保存的模型
    ):

    model_options = load_config(save_to)
    word_vocab = load_dict(vocab)
    word_vocab_idx = {x: i for i, x in enumerate(word_vocab)}

    # 准备测试数据

    test_x = [['I','come']]
    test_x_id = np.array([[word_vocab_idx.get(word,1) for word in test_x[0]]])

    #定义图
    g = Graph(model_options, is_training=False)
    g.build_graph()

    with tf.Session(graph=g.graph) as sess:
        sv = tf.train.Saver()  # 用于保存模型

        feed = {g.x: test_x_id}
        sv.restore(sess,save_to+str(iter))
        test_outputs = sess.run(g.output,feed)
        for data in test_outputs:
            print data

        test_outputs = np.max(test_outputs, 1)
        print word_vocab[test_outputs[0]]



if __name__ == '__main__':
    prob(
        save_to='model/nnlm.ckpt',
        vocab='data/vocab.json',
        iter=2000  # 第几次保存的模型
    )











