from learning_agent import QLearningAgent


class Test(object):

    def __init__(self, verbose=False):
        self.agent = QLearningAgent(verbose=verbose)

    def move(self):
        action = self.agent.get_action()
        (next_state, terminal_state, reward) = self.agent.next_state(action)
        return (action, next_state, reward, terminal_state)

    def update(self, action, next_state, reward):
        self.agent.update_qtable(action, reward, next_state)

    def learn(self, num_of_episodes=10000, max_steps=1000):
        for episode in range(num_of_episodes):
            self.agent.env_reset()
            for step in range(max_steps):
                (a, next_s, r, end) = self.move()
                if not end:
                    self.update(a, next_s, r)
                else:
                    self.update(a, next_s, r)
                    break
        print("QLearning completed")

    def best_move(self):
        action = self.agent.get_best_action()
        (next_state, terminal_state, reward) = self.agent.next_state(action)
        self.agent.current_state = next_state
        return (action, next_state, reward, terminal_state)

    def hunt_the_wumpus(self):
        self.agent.env_reset()
        self.agent.env.display_grid()
        (a, next_s, r, end) = self.best_move()
        total_reward = r
        while not end:
            self.agent.env.display_grid()
            (a, next_s, r, end) = self.best_move()
            total_reward += r
        print(total_reward)
