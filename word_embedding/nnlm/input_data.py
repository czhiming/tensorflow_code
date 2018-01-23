#encoding:utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import json

class TextLoader():
    def __init__(self, input_file, vocab_dir, batch_size, win_size, mini_frq=5):
        self.batch_size = batch_size
        self.win_size = win_size #窗口大小
        self.mini_frq = mini_frq #过滤出现次数少于min_frq的词
        self.vocab = None
        self.vocab_size = 0

        data_dir = os.path.dirname(input_file)
        vocab_file = os.path.join(vocab_dir) #词汇文件,代码生成

        self.preprocess(input_file, vocab_file) #预处理文件
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):#所有句子变成了列表形式
        word_counts = collections.Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent) #更新counter内容
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]

        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            json.dump(self.words, f, indent=2)

        raw_data = [[0] * self.win_size +
            [self.vocab.get(w, 1) for w in line] + #返回键值，如果不存在返回1
            [2] * self.win_size for line in lines]
        self.raw_data = raw_data #前后填充占位符[0,0,0,0,0,....,2,2,2,2,2]

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data:
            for ind in range(self.win_size, len(row)):
                xdata.append(row[ind-self.win_size:ind]) #上下文窗口中的词
                ydata.append([row[ind]]) #目标词
        self.num_batches = int(len(xdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size]) # 最后剩余的会被舍弃
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0) #每个batch都单独成为一个矩阵
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


def test():
    data_dir = '../data/xinhua'
    #data_dir = '../data/tinyshakespeare'
    batch_size = 64
    seq_length = 3
    loader = TextLoader(data_dir, batch_size, seq_length)
    xdata, ydata = loader.next_batch()
    print len(xdata), len(ydata)
    print len(xdata[1]), len(ydata[1])
    print xdata[1]
    print ydata[1]
    for ind in xdata[1]:
        print loader.words[ind]
    for ind in ydata[1]:
        print loader.words[ind]
    #print [loader.words[ind] for ind in xdata[2]]
    #print [loader.words[ind] for ind in ydata[2]]

if __name__ == '__main__':
    test()