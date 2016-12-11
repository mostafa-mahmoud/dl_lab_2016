import numpy as np
import tensorflow as tf

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
# 
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

train_data = trans.get_train()
valid_data = trans.get_valid()

import keras.layers.convolutional.Convolution1D
from keras.optimizers import SGD

inp_shape = (10, 32)

def define_model(inp_shape, neurons=[64, 32]):
  model = Sequential()
  layers = len(neurons)
  model.add(Convolution1D(neurons[0], 3, border_mode='same', input_shape=inp_shape)
  for i in xrange(1, layers):
    model.add(Convolution1D(neurons[i], 3, border_mode='same'))
    model.add(Activation("relu"))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  return model

cnn_astar_mimic = define_model(opt.state_size)

for i in xrange(opt.minibatch_size):
  X_batch, Y_batch = trans.sample_minibatch()
  # yp = cnn_astar_mimic.predict(x)
  # cnn_astar_mimic.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
  cnn_astar_mimic.train_on_batch(X_batch, Y_batch)

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
    

# 2. save your trained model

