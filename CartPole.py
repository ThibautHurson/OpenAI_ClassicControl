import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

'''
A pole is attached by an un-actuated joint to a cart, which moves along a 
frictionless track. The system is controlled by applying a force of +1 or -1 
to the cart. The pendulum starts upright, and the goal is to prevent it from 
falling over. A reward of +1 is provided for every timestep that the pole 
remains upright. The episode ends when the pole is more than 15 degrees from 
vertical, or the cart moves more than 2.4 units from the center.
'''
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'CartPole-v1'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print('Number of actions: ', nb_actions)
print('Observation Space shape: ', env.observation_space.shape)

# Model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(nb_actions))

# DQN
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)#, enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg',)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Training
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

# Testing
dqn.test(env, nb_episodes=5, visualize=True)
