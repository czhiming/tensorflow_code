#encoding:utf-8

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import random
import itertools
import sys
from collections import Counter, defaultdict

class TextLoader():
    def __init__(self, data_dir, batch_size, win_size=5, mini_frq=5):
        self.data_dir = data_dir # 文件路径
        self.batch_size = batch_size # batch的大小
        self.win_size = win_size # chou
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "text8") #输入文件
        vocab_file = os.path.join(data_dir, "vocab.pkl") #词汇文件

        self.preprocess(input_file, vocab_file) #预处理文件
        
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
        vocabulary_inv = ['<UNK>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} #注意此处的书写方式
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        """
        @function: 对文件进行预处理
        """
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            # if lines[0][:1] == codecs.BOM_UTF8:
            #     lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]
        
        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)
        print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        word_counts = Counter()
        cooccurrence_counts = defaultdict(float) #限定值为float
        for words in lines:
            word_ids = [self.vocab.get(w, 0) for w in words]
            word_counts.update(word_ids)
            for id in range(len(word_ids)):
                #计算词词共现
                for c_id in range(max(0, id - self.win_size), id):
                    dist = id - c_id
                    cooccurrence_counts[(word_ids[id], word_ids[c_id])] += 1.0 / (dist + 1)
                for c_id in range(id + 1, min(id + self.win_size + 1, len(word_ids))):
                    dist = c_id - id
                    cooccurrence_counts[(word_ids[id], word_ids[c_id])] += 1.0 / (dist + 1)
        self.cooccurrence_matrix = {(words[0], words[1]): count
                                    for words, count in cooccurrence_counts.items() if count > 10}

    def create_batches(self):
        """
        @function: 创建batch
        """
        row_data, col_data, val_data = list(), list(), list()
        for ((w1, w2), v) in self.cooccurrence_matrix.items():
            row_data.append(w1)
            col_data.append(w2)
            val_data.append(v)
        self.num_batches = int(len(row_data) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        row_data = np.array(row_data[:self.num_batches * self.batch_size])
        col_data = np.array(col_data[:self.num_batches * self.batch_size])
        val_data = np.array(val_data[:self.num_batches * self.batch_size])

        self.row_batches = np.split(row_data, self.num_batches, 0)
        self.col_batches = np.split(col_data, self.num_batches, 0)
        self.val_batches = np.split(val_data, self.num_batches, 0)

    def next_batch(self):
        """
        @function: 创建下一个batch
        """
        rs, cs, vs = self.row_batches[self.pointer], self.col_batches[self.pointer], self.val_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batches
        return rs, cs, vs

    def reset_batch_pointer(self):
        self.pointer = 0


def test():
    data_dir = '../data/xinhua'
    #data_dir = '../data/tinyshakespeare'
    batch_size = 64
    win_size = 3
    loader = TextLoader(data_dir, batch_size, win_size)
    row_data, col_data, val_data = loader.next_batch()
    print len(row_data), len(col_data), len(val_data)
    for ind in row_data:
        print loader.words[ind]
    for ind in col_data:
        print loader.words[ind]
    for ind in val_data:
        print ind

if __name__ == '__main__':
    test()