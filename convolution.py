#!/usr/bin/env python

import time
import glob
import tensorflow as tf
import numpy as np
#from readData import image as dev_images
from create_dataset import getDataset
batch_size = 128
# test_size = 256, full test

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

#trX, trY, teX, teY = dev_images()
start_time = time.time()
trX, trY, teX, teY = getDataset()
print "Reading Dataset Complete, time taken: {}".format(time.time() - start_time)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 104])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 104])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

tf.train.export_meta_graph('')
saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    if False:
        saver.restore(sess, '/Users/nithinvasisth/Documents/advanced_ml/asgn/devnagari/base_model.ckpt')
        print "Model restored: {}".format('base_model.ckpt')

    for i in range(100):
        start_time = time.time()
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        train_time = time.time() - start_time
        start_time = time.time()

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        #test_indices = test_indices[0:test_size]

        save_path = saver.save(sess, "/tmp/model_{}.ckpt".format(i))
        if (i % 5 == 0) and (i>5):
            print "Iterations: {0}, Train Accuracy: {2}, Test Accuracy: {1}".format(
                            i, np.mean(np.argmax(teY[:], axis=1) ==
                            sess.run(predict_op, feed_dict={X: teX[:],
                                                             Y: teY[:],
                                                             p_keep_conv: 1.0,
                                                             p_keep_hidden: 1.0}))
                            , np.mean(np.argmax(trY[:], axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX[:],
                                                             Y: trY[:],
                                                             p_keep_conv: 1.0,
                                                             p_keep_hidden: 1.0})))
            eval_time = time.time() - start_time
            print "Train time: {}, Eval time: {}".format(train_time, eval_time)

