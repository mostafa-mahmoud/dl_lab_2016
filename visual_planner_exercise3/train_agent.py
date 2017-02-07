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

from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense, Flatten, Reshape


def define_model(inp_shape):
  model = Sequential()
  #input Layer
  #model.add(Reshape((1,) + inp_shape, input_shape=inp_shape))
  #model.add(Convolution1D(64, 3, border_mode='same', input_shape=inp_shape))
  model.add(Reshape(inp_shape, input_shape=inp_shape))

  model.add(Dense(64))
  model.add(Activation("relu"))

  #Hidden Layers
  model.add(Dense(128))
  model.add(Activation("relu"))

  model.add(Reshape((1, 128)))

  model.add(Convolution1D(64, 3, border_mode='same'))
  model.add(Activation("relu"))
  
  model.add(Convolution1D(32, 3, border_mode='same'))
  model.add(Activation("relu"))

  model.add(Convolution1D(64, 3, border_mode='same'))
  model.add(Activation("relu"))

  #model.add(Dense(64))
  #model.add(Activation("tanh"))

  #model.add(Dense(64))
  #model.add(Activation("relu"))

  # Output layer
  model.add(Convolution1D(opt.act_num, 3, border_mode='same'))
  model.add(Activation("softmax"))

  model.add(Reshape((opt.act_num,)))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  return model

cnn_astar_mimic = define_model((opt.state_siz * opt.hist_len,))

for _ in xrange(50):
  for i in xrange(opt.n_minibatches):
    X_batch, Y_batch = trans.sample_minibatch()
    cnn_astar_mimic.train_on_batch(X_batch, Y_batch)

  X_val, Y_val = trans.get_valid()
  loss_and_metrics = cnn_astar_mimic.evaluate(X_val, Y_val, batch_size=opt.minibatch_size)
      

# 2. save your trained model

# serialize model to JSON
model_json = cnn_astar_mimic.to_json()
with open(opt.network_fil, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_astar_mimic.save_weights(opt.weights_fil)
print("Saved model to disk")
