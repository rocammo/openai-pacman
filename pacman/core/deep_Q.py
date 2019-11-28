import sys
#import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers

class DeepQ(object):
    def __init__(self, outer):
        self.construct_q_network(outer)

    def construct_q_network(self, outer):

        EPISODES = 5
        env = outer.env
        # env = outer.env.wrappers.Monitor(env, '/tmp/MsPacman-ram-experiment-1',force=True)
        # get size of state and action from environment
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = Agent(state_size, action_size)

        scores, episodes = [], []

        for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            lives = 3
            while not done:
                dead = False
                while not dead:
                    if agent.render:
                        env.render()

                    # get action for the current state and go one step in environment
                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    # save the sample <s, a, r, s'> to the replay memory
                    agent.append_sample(state, action, reward, next_state, done)
                    # every time step do the training
                    agent.train_model()

                    state = next_state
                    score += reward
                    dead = info['ale.lives']<lives
                    lives = info['ale.lives']
                    # if an action make the Pacman dead, then gives penalty of -100
                    reward = reward if not dead else -100

                if done:
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./pacman.png")
                    print("episode:", e, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon)


            # save the model
            if e % 50 == 0:
                agent.model.save_weights("./pacman.h5")




        print('Successfully constructed DDQN (DeepQ) network.')

    def load_network(self, path):
        # self.model = load_model(path)
        print('Successfully loaded DDQN (DeepQ) network.')



class Agent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = True
        self.load_model = False

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

        if self.load_model:
            self.model.load_weights("./pacman.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
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
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


