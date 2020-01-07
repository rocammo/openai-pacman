import numpy as np
import gym
import sys
import pylab
import random

from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers

from pacman.utils.replay_buffer import ReplayBuffer
from pacman.core.deep_Q import DeepQAgent
from pacman.core.duel_Q import DuelQ

# constants
BUFFER_SIZE = 100000

class PacMan:
    def __init__(self, mode):
        self.env = gym.make('MsPacman-ram-v0')
        self.env.reset()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # construct appropiate network based on flags
        print('\033[95m' + 'INFO: Using', mode, 'on MsPacman-ram-v0' + '\033[0m')

        if mode == 'DDQN':
            state_size = self.env.observation_space.shape[0]
            action_size = self.env.action_space.n
            self.agent = DeepQAgent(state_size, action_size)
        elif mode == 'DQN':
            self.algorithm = DuelQ(self)

        # buffer that keeps the last 3 images
        self.process_buffer = []
        # initialize buffer with the first frame
        s1, r1, _, _ = self.env.step(0)
        s2, r2, _, _ = self.env.step(0)
        s3, r3, _, _ = self.env.step(0)
        self.process_buffer = [s1, s2, s3]

    def load_network(self, path):
        print('\033[95m' + 'INFO: Loading network' + '\033[0m')
        print('load_network')
        self.algorithm.load_network(path)

    def convert_process_buffer(self):
        '''
        convert the list of NUM_FRAMES images in the process buffer
        into one training sample
        '''
        black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)) for x in self.process_buffer]
        black_buffer = [x[1:85, :, np.newaxis] for x in black_buffer]

        return np.concatenate(black_buffer, axis=2)

    def train(self, num_frames):
        print('\033[95m' + 'INFO: Training' + '\033[0m')
        EPISODES = 5
        env = self.env
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = self.agent

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
            print('\033[95m' +'INFO: Episode has ended, saving the network into the ./pacman.h5 file.' + '\033[0m')
            if e % 50 == 0:
                agent.model.save_weights("./pacman.h5")

        print('\033[95m' +'INFO: All episodes were run, exiting.' + '\033[0m')

    def simulate(self, path='', save=False):
        print('\033[95m' + 'INFO: Simulating' + '\033[0m')
        print('simulate')

    def calculate_mean(self, num_samples=100):
        print('\033[95m' + 'INFO: Calculating the mean' + '\033[0m')
        print('calculate_mean')
        pass
