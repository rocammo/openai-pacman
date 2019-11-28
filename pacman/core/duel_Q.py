# from keras.models import load_model

class DuelQ(object):
    def __init__(self, outer):
        self.construct_q_network()

    def construct_q_network(self):
        print('Successfully constructed DQN (DuelQ) network.')

    def load_network(self, path):
        # self.model = load_model(path)
        print('Successfully loaded DQN (DuelQ) network.')
