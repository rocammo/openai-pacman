import numpy as np
from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Dense

# constants
NUM_FRAMES = 3
NUM_ACTIONS = 6
DECAY_RATE = 0.99
TAU = 0.01

class DeepQ:
    def __init__(self):
        self.construct_network()

    def construct_network(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(
            4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        self.target_model = Sequential()
        self.target_model.add(Convolution2D(
            32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_model.set_weights(self.model.get_weights())

        print('Successfully constructed DDQN (DeepQ) network.')

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
        print('Successfully saved DDQN (DeepQ) network.')

    def load_network(self, path):
        self.model = load_model(path)
        print('Successfully loaded DDQN (DeepQ) network.')

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()

        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * \
                                        target_model_weights[i]

        self.target_model.set_weights(target_model_weights)
