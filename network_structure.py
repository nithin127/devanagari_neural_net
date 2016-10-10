# ==============================================================================
# Adapted from https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/mnist/mnist.py
# ==============================================================================

"""Builds the Devanagri network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The Devanagri dataset has 104 classes, representing all the different symbols.
NUM_CLASSES = 104
IMAGE_SIZE = 32 # After downscaling by 80%, input is 64x64x1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Probability assignments for dropouts
KEEP_PROB_CONV = 1
KEEP_PROB_HIDD = 1 # We're not implementing dropout in hidden layer in this iteration

def inference(images, conv1_depth, conv2_depth, conv3_depth, 
              hidden1_units, hidden2_units, receptive_field):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Constructing shapes of convolution layers using user input
  SHAPE_1 = [receptive_field, receptive_field, 1, conv1_depth]
  SHAPE_2 = [receptive_field, receptive_field, conv1_depth, conv2_depth]
  SHAPE_3 = [receptive_field, receptive_field, conv2_depth, conv3_depth]
  # Fully connected layers
  SHAPE_4 = [int(conv3_depth * (IMAGE_SIZE / 8) * (IMAGE_SIZE / 8)), hidden1_units]

  # Convolutional Layer 1
  with tf.name_scope('convolution1'):
    weights = tf.Variable(
      tf.truncated_normal(SHAPE_1, stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
      name = 'weights')
    conv1 = tf.nn.relu(tf.nn.conv2d(images, weights,            # conv1 shape=(?, 64, 64, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],             # conv1 shape=(?, 32, 32, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, KEEP_PROB_CONV)

  # Convolutional Layer 2
  with tf.name_scope('convolution2'):
    conv1_units = reduce(lambda x, y: x*y, conv1.get_shape().as_list())
    weights = tf.Variable(
      tf.truncated_normal(SHAPE_2, stddev=1.0 / math.sqrt(float(conv1_units))),
      name = 'weights')
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights,            # conv2 shape=(?, 32, 32, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],             # conv2 shape=(?, 16, 16, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, KEEP_PROB_CONV)

  # Convolutional Layer 3
  with tf.name_scope('convolution3'):
    conv2_units = reduce(lambda x, y: x*y, conv2.get_shape().as_list())
    weights = tf.Variable(
      tf.truncated_normal(SHAPE_3, stddev=1.0 / math.sqrt(float(conv2_units))),
      name = 'weights')
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights,            # conv2 shape=(?, 16, 16, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],             # conv2 shape=(?, 8, 8, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.reshape(conv3, [-1, SHAPE_4[0]])    # reshape to (?, 8192)
    conv3 = tf.nn.dropout(conv3, KEEP_PROB_CONV)

  # Hidden 1
  with tf.name_scope('hidden1'):
    conv3_units = reduce(lambda x, y: x*y, conv3.get_shape().as_list())
    weights = tf.Variable(
        tf.truncated_normal(SHAPE_4,
                            stddev=1.0 / math.sqrt(float(conv3_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(conv3, weights) + biases)
    hidden1 = tf.nn.dropout(hidden1, KEEP_PROB_HIDD)
  '''
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  '''
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden1, weights) + biases
  
  return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))