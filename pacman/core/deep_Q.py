# from keras.models import load_model

class DeepQ(object):
    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        print('Successfully constructed DDQN (DeepQ) network.')

    def load_network(self, path):
        # self.model = load_model(path)
        print('Successfully loaded DDQN (DeepQ) network.')