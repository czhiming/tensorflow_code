#encoding:utf-8

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from six.moves import cPickle

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  plt.figure(figsize=(18, 18))
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)

def read_analogies(eval_data, vocab):
    questions = []
    questions_skipped = 0
    with open(eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):
          continue
        words = line.strip().lower().split(b" ")
        ids = [vocab.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)

word_vecs = np.load('backup/word_vec.npy')
print 'word num:' + str(word_vecs.shape[0])
with open('backup/vocab.pkl', 'rb') as f:
    vocab, words = cPickle.load(f)

#测试例子
test_words = ['world','when','most','also','zero',
              'after','five','China','of','may']
for word in test_words:
    if not vocab.has_key(word):
        continue
    word_vec = word_vecs[vocab.get(word),:]
    sim_mat = np.matmul(word_vecs, word_vec)
    neareast = (-sim_mat).argsort()[1:11]
    neareast_words = [words[id] for id in neareast]
    print '与词<{0}>最相似的前10个词为：'.format(word) + ','.join(neareast_words)

#一：测试question
question_data = '../data/questions-words.txt'
analogy_questions = read_analogies(question_data, vocab)
correct = 0
total = analogy_questions.shape[0]
start = 0
total = 100
while start < total:
  a,b,c,d = analogy_questions[start, :]
  pred_vec = word_vecs[c] - word_vecs[a] + word_vecs[b]
  nearest = np.matmul(word_vecs, pred_vec)
  nearest = (-nearest).argsort()[:4]
  for j in xrange(4):
    if nearest[j] == d:
        correct += 1
        break
    elif nearest[j] in analogy_questions[start,:]:
        continue
    else:
        break
  start += 1
print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))

#二：画图展示
#tmp_ids = np.random.randint(0, analogy_questions.shape[0], size = [20])
word_ids = np.asarray(list(set(analogy_questions[:20].flatten())))
word_embs = word_vecs[word_ids]
labels = [words[i] for i in word_ids]

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(word_embs)
plot_with_labels(low_dim_embs, labels)