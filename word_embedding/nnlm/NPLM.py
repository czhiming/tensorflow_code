from nltk.corpus import brown
import collections
import tensorflow as tf
import numpy as np

vocabulary_size = 16383
batch_size = 128
embedding_size = 30
window = 4
hidden = 100


def built_dataset(words, vocabulary_size):
    # 计数
    counts = [['UNK', -1]]
    counts.extend(collections.Counter(words).most_common(vocabulary_size))

    # 建立词典
    dictionary = {}
    for word, _ in counts:
        dictionary[word] = len(dictionary)

    # 记录数据集索引
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    counts[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, counts, dictionary, reverse_dictionary


data, counts, dictionary, reverse_dictionary = built_dataset(brown.words(), vocabulary_size=vocabulary_size)
data_index = 0


def generate_batch(data, batch_size, window):  # 生成batch降低内存需求
    length_of_data = len(data)
    global data_index
    batch = np.ndarray(shape=(batch_size, window), dtype=np.int32)
    labels = np.zeros(shape=(batch_size, vocabulary_size + 1), dtype=np.float32)
    for i in range(batch_size):
        if data_index < 4:
            data_index = data_index + length_of_data
        batch[i] = [data[(data_index - window) % (length_of_data - 1)],
                    data[(data_index - window + 1) % (length_of_data - 1)],
                    data[(data_index - window + 2) % (length_of_data - 1)],
                    data[(data_index - window + 3) % (length_of_data - 1)]]
        labels[i][data[data_index % (length_of_data - 1)]] = 1
        data_index = (data_index + 1) % (length_of_data - 1)
    return batch, labels


graph = tf.Graph()

with graph.as_default():
    # 输入
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, window])
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size + 1])
    # embedding层
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size + 1, embedding_size], -1.0, 1.0))
    embed = tf.reshape(tf.nn.embedding_lookup(embeddings, train_inputs), (batch_size, embedding_size * window))
    # 第一个隐藏层
    weight1 = tf.Variable(tf.random_uniform(shape=((embedding_size * window), hidden)), dtype=tf.float32)
    biase1 = tf.Variable(tf.random_uniform(shape=(hidden,)), dtype=tf.float32)
    hidden_layer1_output = tf.matmul(embed, weight1) + biase1
    hidden_layer1_output = tf.tanh(hidden_layer1_output)
    # 第二个隐藏层
    weight2 = tf.Variable(tf.random_uniform(shape=(hidden, vocabulary_size + 1)), dtype=tf.float32)
    biase2 = tf.Variable(tf.random_uniform(shape=(vocabulary_size + 1,)), dtype=tf.float32)
    hidden_layer2_output = tf.matmul(hidden_layer1_output, weight2) + biase2
    hidden_layer2_output = tf.nn.softmax(hidden_layer2_output)
    # 正则项
    regulations = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2) + tf.nn.l2_loss(biase1) + tf.nn.l2_loss(biase2)
    # 损失函数
    loss_function = -tf.reduce_mean(
        tf.log(tf.reduce_sum(hidden_layer2_output * train_labels, reduction_indices=1))) + 1e-5 * (regulations)
    # 使用梯度下降优化
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss_function)
    # 初始化,这个步骤必不可少
    init = tf.initialize_all_variables()

num_steps = 100000

with tf.Session(graph=graph) as session:
    init.run()
    saver = tf.train.Saver()
    averager_loss = 0

    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(data, batch_size, 4)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss_function], feed_dict=feed_dict)
        averager_loss += loss_val

        if step % 500 == 0:
            if step > 0:
                averager_loss /= 500.0
                #         saver.save(session, 'NNPLM_model', global_step=step)
            print("Average loss at %s: %s" % (step, averager_loss))
            averager_loss = 0
