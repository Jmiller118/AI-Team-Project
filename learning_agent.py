import numpy as np
from env import Environment

"""
action lookup:
no_action = 0
up = 1
down = 2
left = 3
right = 4
shoot_up = 5
shoot_down = 6
shoot_left = 7
shoot_right = 8
"""


class QLearningAgent(object):

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.6, verbose=True):
        self.env = Environment()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.verbose = verbose
        self.num_actions = 8
        self.qtable = {}
        self.current_state = self.env.detect_nearby()

    def env_reset(self):
        self.env.reset()
        self.current_state = self.env.detect_nearby()
        if self.verbose:
            self.env.display_grid()

    def get_best_action(self):
        if self.current_state not in self.qtable:
            self.qtable[self.current_state] = np.zeros(self.num_actions)

        if self.env.arrow > 0:
            action = np.random.choice([i for i in range(
                self.num_actions) if self.qtable[self.current_state][i] == np.max(self.qtable[self.current_state])])
        else:
            action = np.random.choice([i for i in range(
                self.num_actions - 4) if self.qtable[self.current_state][i] == np.max(self.qtable[self.current_state][:4])])

        return action + 1

    def get_action(self):
        if self.current_state not in self.qtable:
            self.qtable[self.current_state] = np.zeros(self.num_actions)

        if np.random.uniform(0, 1) > self.epsilon:
            if self.env.arrow > 0:
                action = np.random.randint(0, 8)
            else:
                action = np.random.randint(0, 4)
        else:
            if self.env.arrow > 0:
                action = np.random.choice([i for i in range(
                    self.num_actions) if self.qtable[self.current_state][i] == np.max(self.qtable[self.current_state])])
            else:
                action = np.random.choice([i for i in range(
                    self.num_actions - 4) if self.qtable[self.current_state][i] == np.max(self.qtable[self.current_state][:4])])
        return action + 1

    def next_state(self, action):
        next_state, terminal_state, reward = self.env.get_state(action)
        return (next_state, terminal_state, reward)

    def update_qtable(self, action, reward, next_state):
        if self.verbose:
            self.env.display_grid()
        if next_state not in self.qtable:
            self.qtable[next_state] = np.zeros(self.num_actions)
        self.qtable[self.current_state][action-1] = self.qtable[self.current_state][action-1] + (self.alpha * (
            reward + self.gamma * (np.max(self.qtable[next_state])) - self.qtable[self.current_state][action-1]))
        self.current_state = next_state
