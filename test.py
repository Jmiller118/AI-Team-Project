

class Test:
    def __init__(self, env, agent, verbose=False):
        self.env = env
        self.agent = agent
        self.verbose = verbose


    def step(self):
        observe = self.env.detect_nearby()
        action = self.agent.action(observe)

        (reward, stop) = self.env.detect_nearby(action)
        self.agent.reward(obverve, action, reward)

        return (observe, action, reward, stop)

    def epoch(self, num, max_iter):
        total = 0.0

        for n in range(1, num + 1):
            self.agent.reset()
            self.env.reset()

            for i in range(1, max_iter + 1):
                if self.verbose:
                    print("Step {}:".format(i))
                    self.env.display()

                (observe, action, reward, stop) = self.epoch()
                total += reward

                if self.verbose:
                    print(" ->      observe: {}".format(obverve))
                    print(" ->      action: {}".format(action))
                    print(" ->      reward: {}".format(reward))
                    print(" ->      total: {}".format(total))

                    if stop is None:
                        print(" ->      Event: {}".format(stop))

                    print()

                if stop is not None:
                    break
            
            if self.verbose:
                print(" <=> Finished! Epoch: {} <=>".format(n))
                print()

        return total

    def count(x, total):
        if callable(x):
            return [ x() for _ in range(total)]

        else:
            return list(iter(x))


class Parallel:

    def __init__(self, env, agent, num, verbose=False):
        self.environmnets = count(env, num)
        self.agents = count(agent, num)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.finished = [ False for _ in self.environments]

    def episode(self, max_iter):
        reward = []

        for (agent, env) in zip(self.agents, self.environments):
            agent.reset()
            env.reset()
            total = 0

            for i in range(1, max_iter + 1):
                observe = env.detect_nearby()
                action = agent.action(observe)
                (reward, stop) = end.action(action)
                agent.
