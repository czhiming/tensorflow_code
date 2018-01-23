#encoding:utf-8
import sys
from input_data import *
import numpy as np
import tensorflow as tf
import argparse
import time
from nnlm import Graph
from utils import pp


def train(
    input_file='data/train.en',
    save_to='model/nnlm.ckpt',
    save_freq=100,
    disp_freq=10,
    batch_size=60,
    hidden_num=256,
    win_size=5,
    word_dim=500,
    lrate=0.001,
    num_epochs=10,
    grad_clip=10
    ):

    model_options = locals().copy()
    #准备训练数据
    print 'Load data...',
    data_loader = TextLoader(input_file, batch_size, win_size)
    vocab_size = data_loader.vocab_size
    model_options['vocab_size'] = vocab_size

    print 'Done.'
    #定义图
    print 'Build Graph...',
    g = Graph(model_options,is_training=True)
    g.build_graph()
    print 'Done.'
    #优化器
    print 'Start Optimizing...'

    with tf.Session(graph=g.graph) as sess:
        sv = tf.train.Saver()  # 用于保存模型

        tf.global_variables_initializer().run()
        for e in range(model_options['num_epochs']):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x_, y_ = data_loader.next_batch()
                feed = {g.x: x_, g.y: y_}

                train_loss,  _ = sess.run([g.loss, g.optimizer], feed)
                end = time.time()

                #显示结果，和保存模型
                gs = sess.run(g.global_step)
                if gs % disp_freq == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))
                if gs % save_freq == 0:
                    sv.save(sess, save_to+str(gs))

            np.save('model/nnlm_word_embeddings', g.params[pp('ff_input','embeddings')].eval())

    with open(save_to+'.json','w') as fp:
        json.dump(model_options, fp, indent=2)
    print 'Done.'


if __name__ == '__main__':
    train(
        input_file='data/text8',
        save_to='model/nnlm.ckpt',
        save_freq=1000,
        disp_freq=100,
        batch_size=100,
        hidden_num=500,
        win_size=2,
        word_dim=256,
        lrate=0.01,
        num_epochs=10,
        grad_clip=1.
    )










