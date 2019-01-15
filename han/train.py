#coding=utf-8
import json
import sys
import tensorflow as tf
import model
import time
import os
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Data loading params
tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
tf.flags.DEFINE_integer("vocab_size", 147412, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("max_document_len", 100, "max allowed document len")
tf.flags.DEFINE_integer("max_sentence_len", 100, "max allowed sentence len")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 20, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")

FLAGS = tf.flags.FLAGS
TRAIN_FILE = './new_cid_train_demo'
VALID_FILE = './data_demo'
#TEST_FILE = ''


def preprocess(x, y):
    batch_size = len(x)
    document_sizes = np.array([len(doc) for doc in x], dtype=np.int32)
    document_size = document_sizes.max()
    print 'original_document_size: %s'%(document_size),
    document_size = document_size if document_size < FLAGS.max_document_len else FLAGS.max_document_len
    sentence_sizes_ = [[len(sent) for sent in doc] for doc in x]
    sentence_size = max(map(max, sentence_sizes_))
    print 'original_sentence_size: %s'%(sentence_size),
    sentence_size = sentence_size if sentence_size < FLAGS.max_sentence_len else FLAGS.max_sentence_len
    batch_x = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32) # == PAD
    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(x):
        for j, sentence in enumerate(document):
            if j >= document_size:
                continue
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                if k >= sentence_size:
                    continue
                batch_x[i, j, k] = word
    print 'document_size: %s; sentence_size: %s; x.shape: %s' %(document_size, sentence_size, batch_x.shape)
    batch_y = np.zeros(shape = [batch_size,2], dtype=np.int32)
    for i,label in enumerate(y):
        batch_y[i,label] = 1
    return batch_x, batch_y, document_size, sentence_size


def batch_iter(fin_dir, batch_size, num_epochs, shuffle=True, shuffle_fold = 5):
    y = []
    x = []
    with open(fin_dir, 'rb') as f:
        for line in f:
            line = line.strip().split('\t')
            label = int(line[1])
            doc_index = json.loads(line[2])
            y.append(label)
            x.append(doc_index)
    x_len = [len(i) for i in x]
    x_sort_idx = np.argsort(x_len)
    x = np.array(x)
    y = np.array(y)
    x = x[x_sort_idx]
    y = y[x_sort_idx]
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            ori_idx = np.arange(data_size)
            new_idx = np.zeros(data_size, dtype=np.int32)
            fold_count = data_size / shuffle_fold
            for i in range(shuffle_fold):
                start_idx = i*fold_count
                if i == (shuffle_fold - 1):
                    end_idx = data_size
                else:
                    end_idx = (i+1)*fold_count
                new_idx[start_idx:end_idx] = np.random.permutation(ori_idx[start_idx:end_idx])
            shuffled_x = x[new_idx]
            shuffled_y = y[new_idx]
        else:
            shuffled_x = x
            shuffled_y = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_x = shuffled_x[start_index:end_index]
            batch_y = shuffled_y[start_index:end_index]
            batch_x, batch_y, document_size, sentence_size = preprocess(batch_x, batch_y)
            yield batch_x, batch_y, document_size, sentence_size

print "Loading dev data ..."
dev_generator = batch_iter(VALID_FILE, 1000, 1, False)
dev_x, dev_y, document_size_dev, sentence_size_dev = dev_generator.next()
print "Loading dev data finished"


with tf.Session() as sess:
    han = model.HAN(vocab_size=FLAGS.vocab_size,
                    num_classes=FLAGS.num_classes,
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=han.input_y,
                                                                      logits=han.out,
                                                                      name='loss'))
    with tf.name_scope('accuracy'):
        predict = tf.argmax(han.out, axis=1, name='predict')
        label = tf.argmax(han.input_y, axis=1, name='label')
        acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)


    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch, max_sentence_num, max_sentence_length):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: max_sentence_num,
            han.max_sentence_length: max_sentence_length,
            han.batch_size: x_batch.shape[0]
        }
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)

        time_str = str(int(time.time()))
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        train_summary_writer.add_summary(summaries, step)

        return step

    def dev_step(x_batch, y_batch, max_sentence_num, max_sentence_length, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: max_sentence_num,
            han.max_sentence_length: max_sentence_length,
            han.batch_size: x_batch.shape[0]
        }
        step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)
        time_str = str(int(time.time()))
        print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    train_generator = batch_iter(TRAIN_FILE, FLAGS.batch_size, FLAGS.num_epochs, True, 5)
    for batch_idx, train_data in enumerate(train_generator):
        print('current batch %s' % (batch_idx + 1))
        x, y, max_ducument_len, max_sentence_len = train_data
        step = train_step(x, y, max_ducument_len, max_sentence_len)
        if step % FLAGS.evaluate_every == 0:
            dev_step(dev_x, dev_y, document_size_dev, sentence_size_dev, dev_summary_writer)
