from __future__ import division
from __future__ import print_function
import numpy as np

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


class Environment(object):

    def __init__(self, layout, size=(4, 4)):
        # lets make the environment for our agent
        self.layout = layout
        layout_file = open(layout, 'r')
        rows = layout_file.readlines()
        self.wumpus = None
        self.gold_location = None
        self.gold = None
        self.pits = []
        self.start_location = None
        self.wumpus_alive = None
        self.size = size
        y = size[1]-1
        for row in rows[:size[1]]:
            for x, loc in enumerate(row[:size[0]].split(',')):
                if loc == 'W':
                    self.wumpus = (x, y)
                if loc == 'P':
                    self.pits.append((x, y))
                if loc == 'G':
                    self.gold_location = (x, y)
                if loc == 'A':
                    self.start_location = (x, y)
            y -= 1
        self.reset()

    def get_state(self, action):
        """Given action and location, return next state"""
        if action < 4:
            self.agent = self.move(action)
        reward = self.reward(action)
        next_state = self.detect_nearby()
        self.Grab()
        terminal_state = self.terminal_state()
        return next_state, terminal_state, reward

    def terminal_state(self):
        """Check if it's terminal state, Not finished"""
        if self.agent in self.pits or (self.agent == self.wumpus and self.wumpus_alive):
            return True
        if self.agent == self.start_location and not self.gold:
            return True
        return False

    def reset(self):
        """Reset and return init state"""
        # reset for a new training
        self.arrow = True
        self.wumpus_alive = True
        self.gold = True
        self.agent = self.start_location
        state = self.detect_nearby()
        return state

    def Stench(self):
        (i, j) = self.agent
        if self.wumpus_alive and (i - self.wumpus[0]) ** 2 + (j - self.wumpus[1]) ** 2 <= 1:
            return True
        return False

    def Breeze(self):
        (i, j) = self.agent
        for (ii, jj) in self.pits:
            if (i - ii) ** 2 + (j - jj) ** 2 <= 1:
                return True
        return False

    # def Glitter(self):
    #     if self.agent == self.gold_location:
    #         return True
    #     return False

    def detect_nearby(self):
        # if something is nearby it will detect it
        return (self.agent, self.Stench(), self.Breeze(), not self.gold, self.arrow)

    def move(self, act):
        (x, y) = self.agent
        stochastic = np.random.uniform()
        if act == 0:
            if stochastic < 0.8:
                y += 1
            else:
                noise = np.random.choice(2, 1)
                x = (x+1 if noise == 0 else x-1)

        elif act == 1:
            if stochastic < 0.8:
                x += 1
            else:
                noise = np.random.choice(2, 1)
                y = (y+1 if noise == 0 else y-1)

        elif act == 2:
            if stochastic < 0.8:
                y -= 1
            else:
                noise = np.random.choice(2, 1)
                x = (x+1 if noise == 0 else x-1)

        elif act == 3:
            if stochastic < 0.8:
                x -= 1
            else:
                noise = np.random.choice(2, 1)
                y = (y+1 if noise == 0 else y-1)

        x = max(0, min(self.size[0] - 1, x))
        y = max(0, min(self.size[1] - 1, y))
        return x, y

    def Grab(self):
        if self.agent == self.gold_location:
            self.gold = False

    def kill_wumpus(self, loc):
        if loc == self.wumpus:
            self.wumpus_alive = False
            return False
        return True

    def shoot_arrow(self, action):

        (x, y) = self.agent

        if action == 4:
            return self.kill_wumpus((x, y + 1))

        elif action == 5:
            return self.kill_wumpus((x + 1, y))

        elif action == 6:
            return self.kill_wumpus((x, y - 1))

        elif action == 7:
            return self.kill_wumpus((x - 1, y))

    def reward(self, action):

        if self.agent in self.pits:
            return -100

        if self.wumpus_alive and self.agent == self.wumpus:
            return -100

        if action >= 4 and self.arrow:
            self.shoot_arrow(action)
            if not self.wumpus_alive:
                self.arrow = False
                return 10
            else:
                self.arrow = False
                return -100

        if self.agent == self.gold_location and self.gold:
            return 75

        if self.agent == self.start_location and not self.gold:
            return 25

        return -1

    def display_grid(self):
        print("+-", end="")

        for x in range(self.size[0]):
            print("--", end="")

        print("+ ")

        for y in range(self.size[1] - 1, -1, -1):
            print("| ", end="")

            for x in range(self.size[0]):
                if (x, y) == self.wumpus and self.wumpus_alive:
                    print("W ", end='')

                elif (x, y) in self.pits:
                    print("0 ", end='')

                elif (x, y) == self.gold_location and self.gold:
                    print("G ", end='')

                elif (x, y) == self.agent:
                    print("A ", end='')

                else:
                    print(". ", end='')

            print("|")
        print("+", end='')

        for x in range(self.size[0]):
            print("--", end='')
        print("+")
