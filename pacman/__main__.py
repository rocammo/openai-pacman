import argparse

from pacman.instance import PacMan

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Train and test different networks on Pac-Man')
    parser.add_argument('-n', '--network', type=str, action='store', help='Please specify the network you wish to use, either DQN or DDQN', required=True)
    parser.add_argument('-m', '--mode', type=str, action='store', help='Please specify the mode you wish to run, either train or test', required=True)
    parser.add_argument('-p', '--path', type=str, action='store', help='Please specify the directory you wish to render simulation of network in', required=False)
    parser.add_argument('-s', '--statistics', action='store_true', help='Plot statistics of the network', required=False)
    parser.add_argument('-v', '--view', action='store_true', help='Display the network playing the game', required=False)

    args = parser.parse_args()
    print(args)

    game_instance = PacMan(args.network, args.mode, args.view)

    game_instance.train(args.path, args.statistics)