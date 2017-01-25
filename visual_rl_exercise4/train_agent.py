import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from random import randrange
import tensorflow as tf

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense, Flatten, Reshape




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def loss_util(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
       N : minibatch size
       A : number of actions
    Required inputs:
       Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the 
            Q value for each action, each output thus is Q(s,a) for one action 
            (this is easier to implement than adding the action as an additional input to your network)
       action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                      (e.g. each row contains only one 1)
       Q_s_next: a NxA matrix containing the Q values for the next states.
       best_action_next: a NxA matrix with the best current action for the next state
       reward: a Nx1 matrix containing the reward for the transition
       terminal: a Nx1 matrix indicating whether the next state was a terminal state
       discount: the discount factor
    """
    # calculate: reward + discount * Q(s', a*),
    target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    target_q = tf.stop_gradient(target_q)
    return target_q

def Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
       N : minibatch size
       A : number of actions
    Required inputs:
       Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the 
            Q value for each action, each output thus is Q(s,a) for one action 
            (this is easier to implement than adding the action as an additional input to your network)
       action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                      (e.g. each row contains only one 1)
       Q_s_next: a NxA matrix containing the Q values for the next states.
       best_action_next: a NxA matrix with the best current action for the next state
       reward: a Nx1 matrix containing the reward for the transition
       terminal: a Nx1 matrix indicating whether the next state was a terminal state
       discount: the discount factor
    """
    # calculate: reward + discount * Q(s', a*),
    # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
    #target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    #target_q = tf.stop_gradient(target_q)
    target_q = loss_util(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount)
    # calculate: Q(s, a) where a is simply the action taken to get from s to s'
    selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
    loss = tf.reduce_sum(tf.square(selected_q - target_q))    
    return loss

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

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
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# You should prepare your network training here. I suggest to put this into a
# class by itself but in general what you want to do is roughly the following
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# setup placeholders for states (x) actions (u) and rewards and terminal values
x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

# get the output from your network
Q = my_network_forward_pass(x)
Qn =  my_network_forward_pass(xn)

# calculate the loss
loss = Q_loss(Q, u, Qn, ustar, r, term)

# setup an optimizer in tensorflow to minimize the loss
"""

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 10**6
epi_step = 0
nepisodes = 0

cnn_model = define_model(inp_shape=(opt.hist_len*opt.state_siz,)) # TODO check -1 or batchsize

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would let your agent take its action
    #       remember
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # this just gets a random action
    #if random.random() < PR_EPSILON:
     #   pass
        # TODO: Action a maximizes Q(s,a)
        # Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
        # action = np.argmax(np.array([Q_loss(state, _action, next_state_with_history, reward, state.terminal) for _action in range(opt.act_num)]))
        # DEBUG Check ranges [1, N] or [1, N)
    
    action = randrange(opt.act_num)
    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would train your agent
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    state_history_batch, action_batch, next_state_history_batch, reward_batch, terminal_batch = trans.sample_minibatch()
    # cnn_model.train_on_batch(state_batch, next_state_batch) # TODO(done): remove
    # TODO train me here
    # this should proceed as follows:
    # 1)done pre-define variables and networks as outlined above
    # 1) here: calculate best action for next_state_batch
    state_batch = cnn_model.predict(state_history_batch)
    next_state_batch = cnn_model.predict(next_state_history_batch)
    #state_ph, action_ph, next_state_ph, reward_ph, terminal_ph = trans.sample_minibatch()
    # TODO(done)
    # action_batch_next = CALCULATE_ME
    # 2) with that action make an update to the q values
    #    as an example this is how you could print the loss 
    #print(sess.run(loss, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch}))
    #print(next_state_batch)
    #print(np.argmax(next_state_batch, axis=1))
    #for x in map(lambda x: trans.one_hot_action(x), np.argmax(next_state_batch, axis=1).tolist()): print(x)
    action_batch_next = np.array(list(map(lambda act: trans.one_hot_action(act).flatten().tolist(), np.argmax(next_state_batch, axis=1).tolist())))
    q_true = loss_util(state_batch, action_batch, next_state_batch, action_batch_next, reward_batch, terminal_batch)
    print(q_true.__dict__)
    help(q_true)
    loss = cnn_model.train_on_batch(next_state_history_batch, q_true)

    #err = sess.run(loss, feed_dict = {state_ph : state_batch,
    #                                  action_ph : action_batch,
    #                                  best_action_ph : action_batch_next,
    #                                  next_state_ph : next_state_batch,
    #                                  reward_ph : reward_batch,
    #                                  terminal_ph : terminal_batch})

    
    # TODO(done) every once in a while you should test your agent here so that you can track its performance
    if step % 1000 == 0:
        print("Step %d : loss %.3f" % (step, loss))
    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()


# 2. perform a final test of your model and save it
# TODO. done.

# serialize model to JSON
model_json = cnn_model.to_json()
with open(opt.network_fil, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_model.save_weights(opt.weights_fil)
print("Saved model to disk")
