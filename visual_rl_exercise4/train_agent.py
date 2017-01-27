import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from random import randrange, random
import tensorflow as tf

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
    target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    target_q = tf.stop_gradient(target_q)
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



def getNetwork(state_node, batchsize, NUM_LABELS):

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

    assert tf.shape(state_node).eval({})[0] == batchsize
    x_image = tf.reshape(state_node, [tf.shape(state_node).eval({})[0],4,4,-1])

    N0, N1, N2, N3 = 64, 128, 32, 1024
    W_conv0 = weight_variable([5, 5, tf.shape(x_image).eval({})[-1], N0])
    b_conv0 = bias_variable([N0])

    h_conv0 = tf.nn.relu(conv2d(x_image, W_conv0) + b_conv0)
    #h_pool0 = max_pool_2x2(h_conv0)
    #h_pool0 = x_image
    W_conv1 = weight_variable([5, 5, N0, N1])
    b_conv1 = bias_variable([N1])

    #h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(conv2d(h_conv0, W_conv1) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, N1, N2])
    b_conv2 = bias_variable([N2])

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    #h_conv2_flat = tf.reshape(h_conv2, [tf.shape(h_conv1).eval({})[0], -1])#[-1, 8*8*N2])
    h_conv2_flat = tf.reshape(h_conv2, [batchsize, -1])#[-1, 8*8*N2])
    #h_pool2 = max_pool_2x2(h_conv2)
    #h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*N2])
    #print('midsize3', h_pool2_flat)

    #W_fc1 = weight_variable([4*4* N2, N3])
    W_fc1 = weight_variable([tf.shape(h_conv2_flat).eval({})[-1], N3])
    b_fc1 = bias_variable([N3])

    #h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*N2])
    #h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*N2])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    W_fc2 = weight_variable([N3, NUM_LABELS])
    b_fc2 = bias_variable([NUM_LABELS])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


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

sess = tf.InteractiveSession()

inp_ph = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
state_ph = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
action_ph = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
action_best_ph = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
next_state_ph = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
reward_ph = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
terminal_ph = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

Qfeed_ph = getNetwork(state_ph, opt.minibatch_size, opt.act_num)
Qfeed_next_ph = getNetwork(next_state_ph, opt.minibatch_size, opt.act_num)

action_best_ph = tf.one_hot(tf.argmax(Qfeed_next_ph, 1), opt.act_num)

loss = Q_loss(Qfeed_ph, action_ph, Qfeed_next_ph, action_best_ph, reward_ph, terminal_ph)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.initialize_all_variables())
# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 10**6
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
avg_loss, avg_siz = 0, 0
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
    if step and random() < opt.action_epsilon:
        action = tf.argmax(Qfeed_ph, 1).eval({state_ph: state_history_batch})[-1]
    else:
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
    # 1)done here: calculate best action for next_state_batch


    # TODO(done)
    # action_batch_next = CALCULATE_ME
    # 2) with that action make an update to the q values
    #    as an example this is how you could print the loss 
    #print(sess.run(loss, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch}))

    #Qfeed_ph = Qfunc.eval({inp_ph: state_history_batch})
    #Qfeed_next_ph = Qfunc.eval({inp_ph: next_state_history_batch})

    sess.run(train_step, feed_dict = {state_ph : state_history_batch,
                                      action_ph : action_batch,
                                      next_state_ph : next_state_history_batch,
                                      reward_ph : reward_batch,
                                      terminal_ph : terminal_batch})

    err = sess.run(loss, feed_dict={state_ph: state_history_batch,
                                    action_ph : action_batch,
                                    next_state_ph : next_state_history_batch,
                                    reward_ph : reward_batch,
                                    terminal_ph : terminal_batch})

    avg_siz += 1
    avg_loss += err
    if step % 100 == 0:
        print("Step %d : loss %.3f" % (step, avg_loss / avg_siz))
        avg_loss, avg_siz = 0, 0
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
