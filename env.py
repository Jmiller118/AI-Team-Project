from __future__ import division
from __future__ import print_function
import numpy as np
import random
'''
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
'''

class Environment(object):
    actions = []

    def __init__(self, grid=(4, 4), wumpus=1, cave=3, arrow=1):
        # lets make the environment for our agent
        np.random.seed(0)
        self.grid = grid
        self.cases = grid[0] * grid[1]
        self.num_wumpus = wumpus
        self.num_cave = cave
        self.num_arrow = arrow

        random_start = np.random.choice(
            self.cases, self.cases, replace=False)
        self.random_location = [(n % self.grid[1], n // self.grid[1])
                           for n in random_start]

        self.gold_location = self.random_location[0]
        self.cave_location = self.random_location[1: self.num_cave+1]
        self.wumpus_location = self.random_location[self.num_cave+2:self.num_cave+3]
        self.reset()


    def get_state(self,action):
        """Given action and location, return next state"""
        self.action = action
        loc = self.agent
        if action > 4 and self.arrow > 0:
            self.wumpus_alive = self.shoot_arrow(action)
        else:
            self.agent = self.move(loc, action)
        state = self.detect_nearby()
        reward = self.reward(action)
        self.remove_gold()
        terminal_state = self.terminal_state()
        return state,terminal_state,reward

    def evaluate_state(self,action):
        self.action = action
        loc = self.agent
        if action > 4 and self.arrow > 0:
            self.wumpus_alive = self.shoot_arrow(action)
        else:
            self.agent = self.move(loc, action)
        state = self.detect_nearby()
        reward = self.evaluation(action)
        self.remove_gold()
        terminal_state = self.terminal_state()
        return state, terminal_state, reward

    def terminal_state(self):
        End_game_loc = self.cave_location + self.wumpus
        for terminal_loc in End_game_loc:
            if self.agent == terminal_loc:
                return True
        if self.agent == self.location and self.gold == 0:
            return True
        return False

    def reset(self):
        """Reset and return init state"""
        # reset for a new training
        rand = random.randint(self.num_cave+3,self.cases-1)
        self.location = self.random_location[rand]

        self.arrow = self.num_arrow
        self.wumpus_alive = True
        self.wumpus = list(self.wumpus_location)
        self.caves = list(self.cave_location)
        self.gold = self.gold_location
        self.agent = self.location
        state, terminal_state, _ = self.get_state(0)
        state = tuple(state)
        return state, terminal_state

    def wumpus_nearby(self):
        (i, j) = self.agent
        for (ii, jj) in self.wumpus:
            if (i - ii) ** 2 + (j - jj) ** 2 <= 1:
                return True
        return False

    def cave_nearby(self):
        (i, j) = self.agent
        for (ii, jj) in self.caves:
            if (i - ii) ** 2 + (j - jj) ** 2 <= 1:
                return True
        return False

    def gold_underneath(self):
        if self.gold == self.agent:
            return True
        return False

    def detect_nearby(self):
        # if something is nearby it will detect it
        # 2. result_trad_ge
        #return self.agent, self.wumpus_nearby(), self.cave_nearby(), self.arrow, self.gold_exist()
        #1. result_init_p
        return self.agent, self.gold_exist(), self.location
        # 3. result_simp
        #return self.agent, self.gold_exist()

    def move(self, loc, act):
        (x, y) = loc

        if act == 1:
            y += 1

        elif act == 2:
            y -= 1

        elif act == 3:
            x -= 1

        elif act == 4:
            x += 1

        elif act == 0:
            x, y = loc

        x = max(0, min(self.grid[0] - 1, x))
        y = max(0, min(self.grid[1] - 1, y))
        self.agent = x, y
        return x, y

    def bump_wall(self, loc, act):
        (x, y) = loc

        if act == 1:
            y += 1

        elif act == 2:
            y -= 1

        elif act == 3:
            x -= 1

        elif act == 4:
            x += 1

        if x < 0 or x > self.grid[0]:
            return True
        elif y < 0 or y > self.grid[1]:
            return True
    def remove_gold(self):
        if self.agent == self.gold:
            self.gold = 0

    def kill_wumpus(self, loc):
        if loc in self.wumpus:
            self.wumpus.remove(loc)
            self.wumpus_alive = False
            return True
        return False

    def gold_exist(self):
        if self.gold == 0:
            return False
        return True

    def shoot_arrow(self, action):

        (x, y) = self.agent

        if action == 5:
            return self.kill_wumpus((x, y + 1))

        elif action == 6:
            return self.kill_wumpus((x, y - 1))

        elif action == 7:
            return self.kill_wumpus((x - 1, y))

        elif action == 8:
            return self.kill_wumpus((x + 1, y))

    def reward(self, action):

        if self.agent in self.caves:
            return -100

        if self.agent in self.wumpus:
            return -100
        if action > 4 and self.arrow > 0:
            self.arrow -= 1
            return -1

        elif action > 4:
            return -5

        if self.gold_underneath():
            return 5

        if self.agent == self.location and self.gold == 0:
            return 100

        if self.bump_wall(self.agent, self.action):
            return -5

        return -1

    def evaluation(self, action):

        if action > 4 and self.arrow > 0:
            self.arrow -= 1
            return -1

        elif action > 4:
            return -1

        if self.agent == self.location and self.gold == 0:
            return 100

        return -1


    def display_grid(self):
        print("+-", end="")

        for x in range(self.grid[0]):
            print("--", end="")

        print("+ ")

        for y in range(self.grid[1] - 1, -1, -1):
            print("| ", end="")

            for x in range(self.grid[0]):
                if (x, y) in self.wumpus:
                    print("W ", end='')

                elif (x, y) in self.caves:
                    print("0 ", end='')

                elif (x, y) == self.gold:
                    print("G ", end='')

                elif (x, y) == self.agent:
                    print("X ", end='')

                else:
                    print(". ", end='')

            print("|")
        print("+", end='')

        for x in range(self.grid[0]):
            print("--", end='')
        print("+")