# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional rgbd_10 model example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import random
import sys
import time

import numpy
import sklearn
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import input_data


# TODO
# These are some useful constants that you can use in your code.
# Feel free to ignore them or change them.
# TODO 
IMAGE_SIZE = 32
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 32
NUM_EPOCHS = (27999 // BATCH_SIZE) + 1
EVAL_BATCH_SIZE = 1024
EVAL_FREQUENCY = (NUM_EPOCHS - 1) // (5413 // EVAL_BATCH_SIZE) # Number of steps between evaluations.
TEST_BATCH_SIZE = 512
TEST_N_EPOCH = (6321 // TEST_BATCH_SIZE) + 1
# This is where the data gets stored
TRAIN_DIR = 'data'
# HINT:
# if you are working on the computers in the pool and do not want
# to download all the data you can use the pre-loaded data like this:
# TRAIN_DIR = '/home/mllect/data/rgbd'


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  return tf.float32

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def fake_data(num_images, channels):
  """Generate a fake dataset that matches the dimensions of rgbd_10 dataset."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, channels),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels

def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    NUM_CHANNELS = 1
    train_data, train_labels = fake_data(256, NUM_CHANNELS)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    num_epochs = 1
  else:
    if (FLAGS.use_rgbd):
      NUM_CHANNELS = 4
      print('****** RGBD_10 dataset ******') 
      print('* Input: RGB-D              *')
      print('* Channels: 4               *') 
      print('*****************************')
    else:
      NUM_CHANNELS = 3
      print('****** RGBD_10 dataset ******') 
      print('* Input: RGB                *')
      print('* Channels: 3               *') 
      print('*****************************')
    # Load input data
    data_sets = input_data.read_data_sets(TRAIN_DIR, FLAGS.use_rgbd)
    num_epochs = NUM_EPOCHS

    train_data = data_sets.train.images
    train_labels= data_sets.train.labels
    test_data = data_sets.test.images
    test_labels = data_sets.test.labels 
    validation_data = data_sets.validation.images
    validation_labels = data_sets.validation.labels

  train_size = train_labels.shape[0]

  test_this_model_after_training = True

  # TODO:
  # After this you should define your network and train it.
  # Below you find some starting hints. For more info have
  # a look at online tutorials for tensorflow:
  # https://www.tensorflow.org/versions/r0.11/tutorials/index.html
  # Your goal for the exercise will be to train the best network you can
  # please describe why you did chose the network architecture etc. in
  # the one page report, and include some graph / table showing the performance
  # of different network architectures you tried.
  #
  # Your end result should be for RGB-D classification, however, you can
  # also load the dataset with NUM_CHANNELS=3 to only get an RGB version.
  # A good additional experiment to run would be to cmompare how much
  # you can gain by adding the depth channel (how much better the classifier can get)
  # TODO:
  
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(None,))
  #eval_data = tf.placeholder(
  #    data_type(),
  #    shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # TODO
  # define your model here
  # TODO

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
  

  def getNetwork(data_node, labels_node):
    x_image = tf.reshape(data_node, [-1,32,32,3])

    N0, N1, N2 = 64, 128, 32
    W_conv0 = weight_variable([5, 5, 3, N0])
    b_conv0 = bias_variable([N0])

    h_conv0 = tf.nn.relu(conv2d(x_image, W_conv0) + b_conv0)
    h_pool0 = max_pool_2x2(h_conv0)
    #h_pool0 = x_image

    W_conv1 = weight_variable([5, 5, N0, N1])
    b_conv1 = bias_variable([N1])

    h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, N1, N2])
    b_conv2 = bias_variable([N2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([4*4* N2, 1024])
    #W_fc1 = weight_variable([8*8* N2, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*N2])
    #h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*N2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([1024, NUM_LABELS])
    b_fc2 = bias_variable([NUM_LABELS])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    #y_conv = tf.Print(y_conv, [y_conv], message="Haaaallooooo ")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, tf.one_hot(labels_node, NUM_LABELS, on_value=1, off_value=0)))

    # TODO
    # then create an optimizer to train the model
    # HINT: you can use the various optimizers implemented in TensorFlow.
    #       For example, google for: tf.train.AdamOptimizer()
    # TODO
    step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), labels_node)
    return step, correct_prediction


  # TODO
  # Make sure you also define a function for evaluating on the validation
  # set so that you can track performance over time
  # TODO
  train_step, train_correct_prediction = getNetwork(train_data_node, train_labels_node)
  accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))


  #input = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # Create a local session to run the training.

  # TODO
  # Make sure you initialize all variables before starting the tensorflow training
  # TODO
  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())
  

  print("Beginning training...")
  for _ in range(3):
    for i in range(NUM_EPOCHS):
      train_data_batch = train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
      train_labels_batch = train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
      sess.run(train_step, feed_dict={train_data_node: train_data_batch, train_labels_node: train_labels_batch})
      if i % EVAL_FREQUENCY == 0:
        k = int(i // EVAL_FREQUENCY)
        validation_data_batch = validation_data[k*EVAL_BATCH_SIZE:(k+1)*EVAL_BATCH_SIZE]
        validation_labels_batch = validation_labels[k*EVAL_BATCH_SIZE:(k+1)*EVAL_BATCH_SIZE]
        #print("Validation %d to %d of size %d" % (k*EVAL_BATCH_SIZE, (k+1) * EVAL_BATCH_SIZE, len(validation_data_batch)))
        val_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
        print("Validation : ", sess.run(accuracy, feed_dict={train_data_node: validation_data_batch, train_labels_node: validation_labels_batch}))
    train = zip(train_data, train_labels)
    numpy.random.shuffle(train)
    train_data, train_labels = zip(*train)

    evaluation = zip(validation_data, validation_labels)
    numpy.random.shuffle(evaluation)
    validation_data, validation_labels = zip(*evaluation)

  print("Finishing training...")

  # Test trained model
  #test_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
  #print(sess.run(accuracy, feed_dict={train_data_node: test_data, train_labels_node: test_labels}))

  #val_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
  #print(sess.run(accuracy, feed_dict={train_data_node: validation_data, train_labels_node: validation_labels}))

  # Loop through training steps here
  # HINT: always use small batches for training (as in SGD in your last exercise)
  # WARNING: The dataset does contain quite a few images if you want to test something quickly
  #          It might be useful to only train on a random subset!
  # For example use something like :
  # for step in max_steps:
  # Hint: make sure to evaluate your model every once in a while
  # For example like so:
  
  # Finally, after the training! calculate the test result!
  # WARNING: You should never use the test result to optimize
  # your hyperparameters/network architecture, only look at the validation error to avoid
  # overfitting. Only calculate the test error in the very end for your best model!

  if test_this_model_after_training:
    errors = []
    for i in range(TEST_N_EPOCH):
      test_data_batch = test_data[i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE]
      test_labels_batch = test_labels[i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE]
      errors.append(sess.run(accuracy, feed_dict={train_data_node: test_data_batch, train_labels_node: test_labels_batch}))
    print("Testing: ", numpy.mean(errors))

    #val_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
    #print(sess.run(accuracy, feed_dict={train_data_node: validation_data, train_labels_node: validation_labels}))
    #test_data_batch = test_data[k*EVAL_BATCH_SIZE:(k+1)*EVAL_BATCH_SIZE]
    #test_labels_batch = test_labels[k*EVAL_BATCH_SIZE:(k+1)*EVAL_BATCH_SIZE]
    #test_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
    #print("Test : ", sess.run(accuracy, feed_dict={train_data_node: test_data_batch, train_labels_node: test_labels_batch}))
    #test_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
    #print("Test error final : ", sess.run(accuracy, feed_dict={train_data_node: test_data, train_labels_node: test_labels}))
    #print('Test error: {}'.format(test_error))
    #print('Confusion matrix:') 
    # NOTE: the following will require scikit-learn
    #print(confusion_matrix(test_labels, numpy.argmax(eval_in_batches(test_data, sess), 1)))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_rgbd',
      default=False,
      help='Use rgb-d input data (4 channels).',
      action='store_true'
  )
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.'
  )
  FLAGS = parser.parse_args()

  tf.app.run()
