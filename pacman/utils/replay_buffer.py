from collections import deque
import random
import numpy as np

class ReplayBuffer:
    '''
    construct a buffer object that stores the past
    moves and samples a set of subsamples
    '''

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        '''
        add an experience to the buffer
            s: current state,
            a: action, r: reward,
            d: done, s2: next state
        '''

        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def sample(self, batch_size):
        '''
        sample a total of elements equal to batch_size from buffer
        if buffer contains enough elements;
        otherwise, return all elements
            list1 = [1, 2, 3, 4, 5, 6]
            random.sample(list1, 3)
            --
            OUTPUT: [3, 1, 2]
        '''

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # map each experience in batch in batches of
        # [array([s1, ..., sN]), ..., array([s21, ..., s2N])]
        s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))

        return s_batch, a_batch, r_batch, d_batch, s2_batch