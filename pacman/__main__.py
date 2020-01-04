import argparse

from pacman.instance import PacMan

# constants
NUM_FRAMES = 1000000

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Train and test different networks on Pac-Man')
    parser.add_argument('-n', '--network', type=str, action='store', help='Please specify the network you wish to use, either DQN or DDQN', required=True)
    parser.add_argument('-m', '--mode', type=str, action='store', help='Please specify the mode you wish to run, either train or test', required=True)
    parser.add_argument('-l', '--load', type=str, action='store', help='Please specify the file you wish to load weights from (i.e. saved.h5)', required=False)
    parser.add_argument('-s', '--save', type=str, action='store', help='Please specify the folder you wish to render simulation of network in', required=False)
    parser.add_argument('-x', '--statistics', action='store_true', help='Calculate statistics of the network', required=False)
    parser.add_argument('-v', '--view', action='store_true', help='Display the network playing the game (overriden by the -s command)', required=False)

    args = parser.parse_args()
    print(args)

    game_instance = PacMan(args.network)

    if args.load:
        game_instance.load_network(args.load)

    if args.mode.lower() == 'train':
        game_instance.train(NUM_FRAMES)

    if args.statistics:
        stat = game_instance.calculate_mean()
        print('-- STATISTICS')
        print(stat)

    if args.save:
        game_instance.simulate(path=args.save, save=True)
    elif args.view:
        game_instance.simulate()

