#coding:utf-8

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import random

class TextLoader():
    def __init__(self, data_dir, batch_size, skip_window=3, mini_frq=5):
        """
        @function: 构造函数
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mini_frq = mini_frq #过滤出现次数少于min_frq的词
        self.skip_window = skip_window

        input_file = os.path.join(data_dir, "text8") #输入文件
        vocab_file = os.path.join(data_dir, "vocab.pkl") #词汇文件
        #tensor_file = os.path.join(data_dir, "data.npy") #原始数据

        #self.preprocess(input_file, vocab_file, tensor_file) #预处理文件
        self.preprocess(input_file, vocab_file, "") #预处理文件
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        """
        @function: 建立词汇表
        """
        word_counts = collections.Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)
        vocabulary_inv = ['<UNK>'] + [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file):
        """
        @function: 预处理阶段
        """
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]

        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)
        print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump([self.vocab, self.words], f)
        print 'store vocab is over'

        raw_data = [[self.vocab.get(w, 1) for w in line] for line in lines]
        self.raw_data = raw_data

    def create_batches(self):
        """
        @function: 创建一个batch数据
        """
        inputs, targets = list(), list()
        for sent in self.raw_data:
            for ind in range(self.skip_window, len(sent) - self.skip_window):
                context = random.randint(0, self.skip_window - 1)
                inputs.append(sent[ind - context])
                targets.append(sent[ind])
                context = random.randint(0, self.skip_window - 1)
                inputs.append(sent[ind + context])
                targets.append(sent[ind])

        self.num_batches = int(len(inputs) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        inputs = np.array(inputs[:self.num_batches * self.batch_size])
        targets = np.array(targets[:self.num_batches * self.batch_size])

        self.inputs_batches = np.split(inputs, self.num_batches, 0)
        self.targets_batches = np.split(targets, self.num_batches, 0)

    def next_batch(self):
        """
        @function: 下一个 batch 数据
        """
        inputs, targets = self.inputs_batches[self.pointer], self.targets_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batches
        return inputs, targets

    def reset_batch_pointer(self):
        """
        @function: 重置 batch 指针
        """
        self.pointer = 0


def test():
    data_dir = ''
    batch_size = 64
    win_size = 3
    loader = TextLoader(data_dir, batch_size, win_size)
    inputs, targets = loader.next_batch()
    print len(inputs), len(targets)
    for ind in inputs:
        print loader.words[ind],
    print
    for ind in targets:
        print loader.words[ind],

if __name__ == '__main__':
    test()
    
    