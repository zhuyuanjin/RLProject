from model.a2c import *
from network.CartPoleNet import Policy,Value, PVnet
import gym

model = PVnet()
env = gym.make('CartPole-v0').unwrapped
env_test = gym.make('CartPole-v0')
agent = A2CAgent(model=model, gamma=0.9, env=env, test_freq=100, env_test=env_test)
agent.fit_model(5, 10000)