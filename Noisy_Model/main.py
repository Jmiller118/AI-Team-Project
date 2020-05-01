from agent import Agent
from wumpus_env import Environment

env = Environment('test.lay')
agent = Agent(env)

agent.qlearning()
agent.test_env_n()
