from agent import Agent
from env import Environment

env = Environment('test.lay')
agent = Agent(env)

agent.qlearning(epochs=10000)
agent.test_env()
