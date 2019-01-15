#coding=utf-8
import json
import sys
import tensorflow as tf
import model
import time
import os
import numpy as np

# Data loading params
tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
tf.flags.DEFINE_integer("vocab_size", 147412, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("max_document_len", 100, "max allowed document len")
tf.flags.DEFINE_integer("max_sentence_len", 100, "max allowed sentence len")
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")

FLAGS = tf.flags.FLAGS
TRAIN_FILE = './data_demo'
VALID_FILE = './data_demo'
#TEST_FILE = ''

def read_dataset(fin_dir):
    with open(fin_dir, 'rb') as f:
        y = []
        x = []
        for line in f:
            line = line.strip().split('\t')
            label = int(line[1])
            doc_index = json.loads(line[2])
            y.append(label)
            x.append(doc_index)
        y = np.array(y)
        x = np.array(x)
        return x, y

print "Loading data ..."
train_x, train_y = read_dataset(TRAIN_FILE)
dev_x, dev_y = read_dataset(VALID_FILE)

print "Loading data finished"



def batch(inputs, y):
    batch_size = len(inputs)
    
    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()
    document_size = document_size if document_size < FLAGS.max_document_len else FLAGS.max_document_len
     
    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))
    sentence_size = sentence_size if sentence_size < FLAGS.max_sentence_len else FLAGS.max_sentence_len
   
    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32) # == PAD
    
    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document):
            if j >= document_size:
                continue
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                if k >= sentence_size:
                    continue
                b[i, j, k] = word
    print 'document_size: %s; sentence_size: %s; x.shape: %s' %(document_size, sentence_size, b.shape)
    y_np = np.zeros(shape = [batch_size,2], dtype=np.int32)
    for i,label in enumerate(y):
        y_np[i,label] = 1
    return b, y_np, document_size, sentence_size

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
            han.batch_size: FLAGS.batch_size
        }
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)
        #_, step, summaries, cost, accuracy = sess.run([train_summary_op, loss, acc], feed_dict)

        time_str = str(int(time.time()))
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        train_summary_writer.add_summary(summaries, step)

        return step

    def dev_step(x_batch, y_batch, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 30,
            han.max_sentence_length: 30,
            han.batch_size: 64
        }
        step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)
        time_str = str(int(time.time()))
        print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    #for epoch in range(FLAGS.num_epochs):
    for epoch in range(1):
        print('current epoch %s' % (epoch + 1))
        #for i in range(0, 200000, FLAGS.batch_size):
        if True:
            #x = train_x[i:i + FLAGS.batch_size]
           # y = train_y[i:i + FLAGS.batch_size]
            x = train_x[:FLAGS.batch_size]
            y = train_y[:FLAGS.batch_size]
            x, y, max_ducument_len, max_sentence_len = batch(x, y)
            #x, y = batch(x, y)
            #sys.exit(0)
            step = train_step(x, y, max_ducument_len, max_sentence_len)
            #if step % FLAGS.evaluate_every == 0:
            #    dev_step(dev_x, dev_y, dev_summary_writer)
