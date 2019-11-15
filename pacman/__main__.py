import argparse

from pacman.instance import PacMan

# Hyperparameters
NUM_FRAMES = 1000000

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Train and test different networks on Pac-Man')
    parser.add_argument('-n', '--network', type=str, action='store', help='Please specify the network you wish to use, either DQN or DDQN', required=True)
    parser.add_argument('-m', '--mode', type=str, action='store', help='Please specify the mode you wish to run, either train or test', required=True)

    args = parser.parse_args()
    print(args)

    game_instance = PacMan(args.network)

    if args.mode == "train":
        game_instance.train(NUM_FRAMES)
