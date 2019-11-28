import numpy as np
import gym

from pacman.core.deep_Q import DeepQ
from pacman.core.duel_Q import DuelQ

class PacMan(object):
    def __init__(self, mode):
        self.env = gym.make('MsPacman-ram-v0')
        self.env.reset()

        # construct appropiate network based on flags
        print(mode)
        if mode == 'DDQN':
            self.algorithm = DeepQ(self)
        elif mode == 'DQN':
            self.algorithm = DuelQ(self)

    def load_network(self, path):
        self.algorithm.load_network(path)

    def train(self, num_frames):
        pass
