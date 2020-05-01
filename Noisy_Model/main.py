from agent import Agent
from wumpus_env import Environment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('layout', type=str, nargs='?', help='path to layout file')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
parser.add_argument('--epochs', type=int, default=100000,
                    help='number of epochs')
parser.add_argument('--verbose', default=False, help='verbose logging', action='store_true')

args = parser.parse_args()

env = Environment(args.layout)
agent = Agent(env, args.verbose)
agent.qlearning(args.alpha, args.gamma, args.epsilon, args.epochs)
agent.test_env_n()
