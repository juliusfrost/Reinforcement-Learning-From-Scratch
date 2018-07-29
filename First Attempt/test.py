'''
A random Reinforcement Learning algorithm mixed from a group of techniques,
made from scratch based on what I remember from David Silver's RL lectures.
Definitely not the best RL algorithm, but shows what I learned without much practice.
At the time of writing this, I have not yet completed the series.

Julius Frost
'''

import numpy as np


states = 4
actions = 4
state_values = np.zeros(states)
action_values = np.zeros((states, actions))
n_values = np.zeros((states, actions))

policy = np.zeros((states, actions))

# This will make the last action the best for all states
rewards = np.arange(states * actions)
rewards = np.reshape(rewards, (states, actions))


def reward_function(state, action):
    assert(0 <= state and state < states)
    assert(0 <= action and action < actions)
    assert(np.shape(rewards) == (states, actions))
    return rewards[state,action]

def step(state, action):
    next_state = (state + 1) % states # Environment's effect on agent
    reward = reward_function(state, action) # Reward from environment
    return (next_state, reward)
    

history = []
state = np.random.randint(0, high = states)
action = np.random.randint(0, high = actions)

alpha = 10

epsilon = 0.3

gamma = 0.99

iterations = 1000

# loop
for i in range(iterations):
    # e-greeedy method
    # pick the action with the highest value, but with a small probability pick an action at random
    action = np.argmax(action_values[state]) 
    if (np.random.rand() < epsilon): 
        action = np.random.randint(0, high = actions)
    
    #n_values[state, action] += 1
    prev_state = state
    # step / commense action and recieve feedback from environment
    state, reward = step(state, action)
    # update action_values
    action_values[prev_state, action] += 1/(alpha) * (reward - action_values[prev_state, action]) * gamma
    #append to history
    history.append((i, prev_state, action, reward))
    
show_history = True
if show_history:
    for iteration, state, action, reward in history:
        print('iteration: ' + str(iteration) + ' state: ' + str(state) + ' action: ' + str(action) + ' reward: ' + str(reward))

#goal is to make these two look as similar as possible
print(rewards)
print(action_values)


state = 0

for i in range(states):
    # only greedy this time with updated action_values
    action = np.argmax(action_values[state]) 
    
    prev_state = state
    # step / commense action and recieve feedback from environment
    state, reward = step(state, action)
    # dont update action_values this time
    #action_values[prev_state, action] += 1/(alpha) * (reward - action_values[prev_state, action]) * gamma
    #append to history
    history.append((i, prev_state, action, reward))
    print('iteration: ' + str(i) + ' state: ' + str(prev_state) + ' action: ' + str(action) + ' reward: ' + str(reward))