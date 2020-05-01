import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

'''
Actions:
Move Up = 0
Move Right = 1
Move Down = 2
Move Left = 3
Shoot Up = 4
Shoot Right = 5
Shoot Down = 6
Shoot Left = 7
'''


class Agent(object):

    def __init__(self, env):
        self.env = env
        self.N_ACTIONS = 8
        self.qtable = {}

    def reset(self):
        self.env.reset()

    def load_qtable(self, filepath):
        self.qtable = pickle.load(open(filepath), 'rb')

    def qlearning(self, alpha=0.2, gamma=0.9, epsilon=0.4, epochs=100000, max_moves=100):
        plot_rewards = []
        for epoch in range(1, epochs+1):
            state = self.env.reset()
            if state not in self.qtable:
                self.qtable[state] = np.zeros(self.N_ACTIONS)
            num_moves = 0
            total_reward = 0
            end = False
            action = None
            while not end and num_moves < max_moves:
                if np.random.uniform() > epsilon:
                    if state[-1]:
                        action = np.random.choice([i for i in range(
                            self.N_ACTIONS) if self.qtable[state][i] == np.max(self.qtable[state])])
                    else:
                        action = np.random.choice([i for i in range(
                            self.N_ACTIONS-4) if self.qtable[state][i] == np.max(self.qtable[state][:self.N_ACTIONS-4])])
                else:
                    if state[-1]:
                        action = np.random.randint(0, self.N_ACTIONS)
                    else:
                        action = np.random.randint(0, self.N_ACTIONS-4)
                next_state, end, reward = self.env.get_state(action)
                if next_state not in self.qtable:
                    self.qtable[next_state] = np.zeros(self.N_ACTIONS)
                self.qtable[state][action] += alpha*(reward+gamma*(
                    np.max(self.qtable[next_state]))-self.qtable[state][action])
                state = next_state
                total_reward += reward
                num_moves += 1

            if epoch % 100 == 0:
                plot_rewards.append(self.test_env())

            if epoch % 1000 == 0:
                print('qlearning {}%'.format((epoch*100)/epochs))

            if epoch % 20000 == 0:
                plt.ioff()
                yhat = savgol_filter(plot_rewards, 51, 3)
                plt.plot(plot_rewards)
                plt.show()
        print('qlearning done')
        # for x in range(self.env.size[0]):
        #     for y in range(self.env.size[1]):
        #         for stench in [True, False]:
        #             for breeze in [True, False]:
        #                 for gold in [True, False]:
        #                     for arrow in [True, False]:
        #                         state = ((x, y), stench, breeze, gold, arrow)
        #                         if ((x, y), stench, breeze, gold, arrow) in self.qtable:
        #                             if arrow:
        #                                 print('{} : {}'.format(
        #                                     state, np.argmax(self.qtable[state])))
        #                             else:
        #                                 print('{} : {}'.format(state, np.argmax(
        #                                     self.qtable[state][:self.N_ACTIONS-4])))
        pickle.dump(self.qtable, open('qtable.pkl', 'wb'))

    def test_env(self, max_moves=100):
        state = self.env.reset()
        if state not in self.qtable:
            self.qtable[state] = np.zeros(self.N_ACTIONS)
        num_moves = 0
        total_reward = 0
        end = False
        action = None
        while not end and num_moves < max_moves:
            # self.env.display_grid()
            if state[-1]:
                action = np.argmax(self.qtable[state])
            else:
                action = np.argmax(self.qtable[state][:self.N_ACTIONS-4])
            # print(action)
            next_state, end, reward = self.env.get_state(action)
            if next_state not in self.qtable:
                self.qtable[next_state] = np.zeros(self.N_ACTIONS)
            state = next_state
            total_reward += reward
            num_moves += 1
        # self.env.display_grid()
        # print(total_reward)
        # raw_input()
        return total_reward

    # run multiple times
    def test_env_n(self, games=100):
        rewards = []
        for game in range(games):
            rewards.append(self.test_env())
        avg_reward = sum(rewards)/len(rewards)
        print(avg_reward)
