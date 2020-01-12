import sys
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers

class DeepQAgent:
    def __init__(self, state_size, action_size, load_model=False):
        print('\033[94m' + 'INFO: DeepQAgent is initializing' + '\033[0m')
        # if you want to see MsPacman learning, then change to True
        self.render = True

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1
        self.batch_size = 128
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model
        self.model = self.build_model()

        if load_model:
            print('\033[94m' + 'INFO: DeepQAgent is loading weights from filesystem' + '\033[0m')
            self.model.load_weights("./results/pacman.h5")
            print('\033[94m' + 'INFO: DeepQAgent sucessfully loaded weights' + '\033[0m')

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        print('\033[94m' + 'INFO: DeepQAgent is building the model' + '\033[0m')
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print('\033[94m' + 'INFO: DeepQAgent model was sucessfully built' + '\033[0m')
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        #print('\033[94m' + 'INFO: DeepQAgent is getting an action' + '\033[0m')
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #print('\033[94m' + 'INFO: DeepQAgent is appending a sample' + '\033[0m')
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        #print('\033[94m' + 'INFO: DeepQAgent is training the model' + '\033[0m')
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)


        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        #print('\033[94m' + 'INFO: DeepQAgent is fitting the model' + '\033[0m')
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
