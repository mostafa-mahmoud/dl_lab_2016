import numpy as np
import matplotlib.pyplot as plt
from random import randrange
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
from keras.models import model_from_json

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# FIXME Check if needed
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# TODO: load your agent
agent = model_from_json(open(opt.network_fil, 'r').read())
agent.load_weights(opt.weights_fil)
print('Loaded model from disk')

# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 0   # total #episodes executed
nepisodes_solved = 0
action = 0     # action to take given by the network

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
for step in range(opt.eval_steps):

    # check if episode ended
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would let your agent take its action
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # this just gets a random action
        #print state.pob.shape
        #print rgb2gray(state.pob).shape

        current_state = rgb2gray(state.pob).reshape(1, opt.state_siz)
        trans.add_recent(epi_step, current_state)
        X_test = trans.get_recent()
        action = agent.predict(X_test, opt.minibatch_size)
        action = np.argmax(action)
        state = sim.step(action)
        epi_step += 1

    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)

    if step % opt.prog_freq == 0:
        print(step)

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

# 2. calculate statistics
print(float(nepisodes_solved) / float(nepisodes))
# 3. TODO perhaps  do some additional analysis
