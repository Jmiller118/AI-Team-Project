import numpy as np

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

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.8):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = 8
        self.qtable = {}

    def get_action(self, state):
        if state not in self.qtable:
            self.qtable[state] = np.zeros(self.num_actions)

        if np.random.uniform(0, 1) > self.epsilon:
            action = np.random.randint(0, 8)
        else:
            action = np.argmax(self.qtable[state])
        return action

    def update_qtable(self, state, action, reward, next_state):
        if next_state not in self.qtable:
            self.qtable[next_state] = np.zeros(self.num_actions)
        self.qtable[state][action] = self.qtable[state][action] + (self.alpha * (
            reward + self.gamma * (np.max(self.qtable[next_state])) - self.qtable[state][action]))
