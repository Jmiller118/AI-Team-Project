from env import Environment
env = Environment()
# env.reset => reset the env and return init state
# env.get_state(action) => given the action, return: (next state, if it's terminal state, reward)
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
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
# init q_table
q_table = {}

# hyper parameter for training
alpha = 0.1
gamma = 0.9
num_itr = 100000
N_ACTIONS = 8
# stochastic move possibility
epsilon = 1
plot_reward = []
input_method = 'manuel1'
max_num_action = 100
env.display_grid()
for i in range(1,num_itr):
    #print("------------------{}--------------------------".format(i))
    state,terminal_state = env.reset()  # get new env and init_state
    if state not in q_table:
        q_table[state] = np.zeros((N_ACTIONS,))  # init q_table
    total_reward = 0
    num_action = 0
    while not terminal_state and num_action < max_num_action:  # terminal_state == True when the agent fall into pit, got the gold sth like that
        if np.random.uniform(0, 1) > epsilon:
            action = np.random.randint(0,8)  # choose a random action
        else:
            action = np.argmax(q_table[state])+1  # choose the optimal action
        #env.display_grid()
        if input_method == 'manuel':
            action = input()
        #time.sleep(.5)
        # get next state, evaluate if it is a terminal state, and get reward(got to somehow design the reward)
        location = state[0]
        #print(location)
        next_state, terminal_state, reward = env.get_state(action)

        if next_state not in q_table:
            q_table[next_state] = np.zeros((N_ACTIONS,))   # init q_table

        old_value = q_table[state][action-1]  # get current q value
        max_action = np.max(q_table[next_state])  # get the best possible future reward
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * max_action)  # based on Lecture 16 Page 23

        q_table[state][action-1] = new_value  # update q_table
        state = next_state  # update current state and continue the loop
        total_reward += reward
        #print(total_reward)
        num_action += 1
    plot_reward.append(total_reward)

    if i % 100 == 0:
        yhat = savgol_filter(plot_reward, 51, 3)
        plt.plot(plot_reward)
        plt.show()
    # Dump it into a pickle file to save the training result
    # Warning: Rerun the code after training complete will rewrite the previous training result
    pickle_out = open("q_table.pkl","wb")
    pickle.dump(q_table, pickle_out,protocol=2)
    pickle_out.close()
print("Training finished.")







