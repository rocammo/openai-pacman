import numpy as np
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Flatten, Dense
from keras.layers import Input, merge
from keras import backend as K

# constants
NUM_FRAMES = 3
NUM_ACTIONS = 6
DECAY_RATE = 0.99

class DuelQ:
    def __init__(self):
        self.construct_network()

    def construct_network(self):
        self.model = Sequential()
        input_layer = Input(shape=(84, 84, NUM_FRAMES))
        conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(input_layer)
        conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv1)
        conv3 = Convolution2D(64, 3, 3, activation='relu')(conv2)
        flatten = Flatten()(conv3)
        fc1 = Dense(512)(flatten)
        advantage = Dense(NUM_ACTIONS)(fc1)
        fc2 = Dense(512)(flatten)
        value = Dense(1)(fc2)
        policy = merge([advantage, value], mode=lambda x: x[0] - K.mean(x[0]) + x[1], output_shape=(NUM_ACTIONS,))
        # policy = Dense(NUM_ACTIONS)(merge_layer)

        self.model = Model(input=[input_layer], output=[policy])
        self.model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        self.target_model = Model(input=[input_layer], output=[policy])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        print('Successfully constructed DQN (DuelQ) network.')

    def predict_movement(self, data, epsilon):
        '''
        predict movement of game controller where is epsilon probability
        randomly moved
        '''
        q_actions = self.model.predict(
            data.reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random_sample()

        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)

        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        '''
        train network to fit given parameters
        '''
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            targets[i] = self.model.predict(
                s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
            fut_action = self.target_model.predict(
                s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # print the loss every 10 iterations
        if observation_num % 10 == 0:
            print('Log: loss equal to', loss)

    def save_network(self, path):
        self.model.save(path)
        print('Successfully saved DQN (DuelQ) network.')

    def load_network(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print('Successfully loaded DQN (DuelQ) network.')

    def target_train(self):
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)