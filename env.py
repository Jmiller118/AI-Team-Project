from __future__ import division
from __future__ import print_function
import numpy as np
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
        self.grid = grid
        self.cases = grid[0] * grid[1]
        self.num_wumpus = wumpus
        self.num_cave = cave
        self.num_arrow = arrow

        random_start = np.random.choice(
            self.cases, self.num_wumpus + self.num_cave + 2, replace=False)
        random_location = [(n % self.grid[1], n // self.grid[1])
                           for n in random_start]

        self.location = random_location[-1]
        self.gold_location = random_location[-2]
        self.cave_location = random_location[0: self.num_cave]
        self.wumpus_location = random_location[self.num_cave:-2]
        self.wumpus = True
        self.reset()

    def get_state(self, action):
        """Given action and location, return next state"""
        loc = self.agent
        if action < 5:
            self.agent = self.move(loc, action)
        self.remove_gold()
        reward = self.reward(action)
        terminal_state = self.terminal_state()
        next_state = self.detect_nearby()
        return next_state, terminal_state, reward

    def terminal_state(self):
        """Check if it's terminal state"""
        End_game_loc = self.cave_location + self.wumpuses
        for terminal_loc in End_game_loc:
            if self.agent == terminal_loc:
                return True
        if self.agent == self.gold_location and self.gold == 0:
            return True
        return False

    def reset(self):
        """Reset and return init state"""
        # reset for a new training
        self.arrow = self.num_arrow
        self.wumpus_alive = self.wumpus
        self.wumpuses = list(self.wumpus_location)
        self.caves = list(self.cave_location)
        self.gold = 1
        self.agent = self.location
        state, terminal_state, _ = self.get_state(0)
        state = tuple(state)
        return state, terminal_state

    def wumpus_nearby(self):
        (i, j) = self.agent
        for (ii, jj) in self.wumpuses:
            if (i - ii) ** 2 + (j - jj) ** 2 <= 1:
                return True
        return False

    def cave_nearby(self):
        (i, j) = self.agent
        for (ii, jj) in self.caves:
            if (i - ii) ** 2 + (j - jj) ** 2 <= 1:
                return True
        return False

    # def gold_underneath(self):
    #     if self.gold == self.agent:
    #         return True
    #     return False

    def detect_nearby(self):
        # if something is nearby it will detect it
        return (self.agent, self.wumpus_nearby(), self.cave_nearby())

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

    def remove_gold(self):
        if self.agent == self.gold_location:
            self.gold = 0

    def kill_wumpus(self, loc):
        if loc in self.wumpuses:
            self.wumpuses.remove(loc)
            if len(self.wumpuses) == 0:
                self.wumpus_alive = False
            return True
        return False

    def shoot_arrow(self, action):

        (x, y) = self.agent

        if action == 5:
            self.arrow -= 1
            return self.kill_wumpus((x, y + 1))

        elif action == 6:
            self.arrow -= 1
            return self.kill_wumpus((x, y - 1))

        elif action == 7:
            self.arrow -= 1
            return self.kill_wumpus((x - 1, y))

        elif action == 8:
            self.arrow -= 1
            return self.kill_wumpus((x + 1, y))

    def reward(self, action):

        if self.agent in self.caves:
            return -1000

        if self.agent in self.wumpuses:
            return -1000

        if action > 4:
            if self.shoot_arrow(action):
                return 10
            else:
                return -10

        if self.agent == self.gold_location:
            return 100

        return -1.0

    def display_grid(self):
        print("+-", end="")

        for x in range(self.grid[0]):
            print("--", end="")

        print("+ ")

        for y in range(self.grid[1] - 1, -1, -1):
            print("| ", end="")

            for x in range(self.grid[0]):
                if (x, y) in self.wumpuses:
                    print("W ", end='')

                elif (x, y) in self.caves:
                    print("0 ", end='')

                elif (x, y) == self.gold_location:
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
