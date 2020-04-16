'''
action lookup:
up = 1
down = 2
left = 3
right = 4
shoot_up = 5
shoot_down = 6
shoot_left = 7
shoot_right = 8
'''
# import env
# env.reset => reset the env and return init state
# env.get_state(action) => given the action, return: (next state, if it's terminal state, reward)
import numpy

# init q_table
q_table = {}

# hyper parameter for training
alpha = 0.1
gamma = 0.9
num_itr = 1000

# stochastic move possibility
epsilon = 0.8

for i in range(1,num_itr):
    state = env.reset()  # get new env and init_state
    if state not in q_table:
        q_table[state] = np.zeros((N_ACTIONS,))  # init q_table

    while not terminal_state:  # terminal_state == True when the agent fall into pit, got the gold sth like that
        if random.uniform(0, 1) > epsilon:
            action = np.random.randint(1,8)  # choose a random action
        else:
            action = np.argmax(q_table[state])  # choose the optimal action

        # get next state, evaluate if it is a terminal state, and get reward(got to somehow design the reward)
        next_state, terminal_state, reward = env.get_state(action)

        if next_state not in q_table:
            q_table[next_state] = np.zeros((N_ACTIONS,))   # init q_table

        old_value = q_table[state][action]  # get current q value
        max_action = np.max(q_table[next_state])  # get the best possible future reward
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * max_action)  # based on Lecture 16 Page 23

        q_table[state][action] = new_value  # update q_table
        state = next_state  # update current state and continue the loop

    # Dump it into a pickle file to save the training result
    # Warning: Rerun the code after training complete will rewrite the previous training result
    pickle_out = open("q_table.pkl","wb")
    pickle.dump(q_table, pickle_out,protocol=2)
    pickle_out.close()

print("Training finished.")







