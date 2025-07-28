

import gym
import random
import numpy as np 


# Initialize the environment
environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

# Define the number of states, actions and qtable
nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states ,nb_actions))


print("Q-Table")
print(qtable)

"""
left: 0
down: 1
right: 2
up: 3

"""


action = environment.action_space.sample()

# New state according to the action
new_state, reward, done, info, _ = environment.step(action) 


# %% Training Part 


import gym
import random
import numpy as np 
import matplotlib.pyplot as plt


from tqdm import tqdm

# Initialize the environment
environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

# Define the number of states, actions and qtable
nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states ,nb_actions))


print("Q-Table")
print(qtable) # Brain of the agent

episodes = 1000 # episode
alpha = 0.5 # learning rate
gamma = 0.9 # discount factor

outcomes = []

# training
for _ in tqdm(range(episodes)):
    state, _ = environment.reset()
    done = False # Success rate of the agent
    
    outcomes.append("Failure")
    
    while not done: # Till agent will be successful
        # action    
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])

        else:
            action = environment.action_space.sample()

        new_state, reward, done, info, _ = environment.step(action) 

        # update Q-table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        state = new_state
        
        if reward:
            outcomes[-1] = "Success"
                

plt.bar(range(episodes), outcomes)


# %% Test Part


import gym
import random
import numpy as np 
import matplotlib.pyplot as plt


# from tqdm import tqdm

episodes = 100 # episode
nb_success = 0



# training
for _ in tqdm(range(episodes)):
    state, _ = environment.reset()
    done = False # Success rate of the agent
    
    
    while not done: # Till agent will be successful
        # action    
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])

        else:
            action = environment.action_space.sample()

        new_state, reward, done, info, _ = environment.step(action) 
        
        state = new_state
        
        nb_success += reward


print("Success Rate: ", 100 * nb_success / episodes)










































