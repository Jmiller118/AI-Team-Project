from env import Environment
env = Environment()
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
import random
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
# hyper parameter for training

class qlearning(object):
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        #self.num_itr = 50000000
        self.num_itr = 10000000
        self.N_ACTIONS = 8
        # stochastic move possibility
        self.epsilon = 0.8
        self.plot_reward = []
        self.max_num_action = 100
        self.input_method = 'AUTO'
        self.test_mode = False
        if not 1:
            self.input_method = 'manuel'
            self.test_mode = True

    def init_q_table(self,state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((self.N_ACTIONS,))  # init q_table

    def action_func(self,q_table, epsilon,state):
        if np.random.uniform(0, 1) > epsilon:
              # choose a random action
            action = np.argmax(q_table[state]) + 1
            noisy_action = random.randint(1, 8)
            next_state, terminal_state, reward = env.get_state(noisy_action)
        else:
            if env.arrow > 0:
                action = np.argmax(q_table[state]) + 1  # choose the optimal action
            else:
                action = np.argmax(q_table[state][0:4] + 1)
            next_state, terminal_state, reward = env.get_state(action)

        return action, next_state, terminal_state, reward

    def update_q(self,q_table,action,state,next_state,reward):
        old_value = q_table[state][action - 1]  # get current q value
        max_action = np.max(q_table[next_state])  # get the best possible future reward
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * max_action)  # based on Lecture 16 Page 23
        q_table[state][action - 1] = new_value  # update q_table


    def data_collection(self,i,plot_reward):
        plt.plot(plot_reward)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Total Reward')
        plt.savefig('epoch_{}.png'.format(i))
        plt.clf()
        yhat = savgol_filter(plot_reward, 201, 3)
        plt.plot(yhat)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Total Reward')
        plt.savefig('epoch_smooth_{}.png'.format(i))
        pickle_out = open("q_table_{}.pkl".format(i), "wb")
        pickle.dump(self.q_table, pickle_out, protocol=2)
        pickle_out.close()



    def main(self):
        env.display_grid()
        for i in range(1,self.num_itr):
            #print("------------------{}--------------------------".format(i))
            #random_seed = random.randint(95, 105)
            #env.display_grid()
            state,terminal_state = env.reset()  # get new env and init_state

            self.init_q_table(state)
            total_reward = 0
            num_action = 0
            while not terminal_state and num_action < self.max_num_action:

                action,next_state,terminal_state,reward = self.action_func(self.q_table, self.epsilon,state)

                if self.input_method == 'manuel':
                    action = input()
                    next_state, terminal_state, reward = env.get_state(action)

                self.init_q_table(next_state)
                self.update_q(self.q_table, action, state, next_state,reward)
                state = next_state  # update current state and continue the loop
                total_reward += reward
                num_action += 1

                if self.test_mode == True:
                    env.display_grid()
                    time.sleep(.5)
                    print(total_reward)

            self.plot_reward.append(total_reward)
            if i % 200000 == 0:
                self.data_collection(i, self.plot_reward)
                print(i)
        print("Training finished.")

if __name__ == "__main__":
    qlearning().main()




