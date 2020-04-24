from env import Environment
from learning_agent import QLearningAgent


class Test(object):

    def __init__(self, verbose=False):
        self.env = Environment()
        self.agent = QLearningAgent()

    def move(self):
        state = self.env.detect_nearby()
        action = self.agent.get_action(state)
        (next_state, terminal_state, reward) = self.env.get_state(action)
        return (state, action, next_state, reward, terminal_state)

    def update(self, state, action, next_state, reward, terminal_state):
        if not terminal_state:
            self.agent.update_qtable(state, action, next_state, reward)

    def learn(self, num_of_episodes=1000, max_steps=1000):
        for episode in range(num_of_episodes):
            self.env.reset()
            for step in range(max_steps):
                (s, a, next_s, r, end) = self.move()
                if not end:
                    self.update(s, a, next_s, r)

    def hunt_the_wumpus(self):
        self.env.reset()
        (s, a, next_s, r, end) = self.move()
        total_reward = r
        if not end:
            (s, a, next_s, r, end) = self.move()
            total_reward += r
        print(total_reward)
