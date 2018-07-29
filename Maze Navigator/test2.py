import numpy as np

#states will be a tuple like (0, 2)
states = {}


actions = ['up', 'down', 'left', 'right']



def create_world():
    grid = np.array([[0,1,0,0,0,0],
                     [0,0,0,1,0,1],
                     [0,1,1,0,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,1,2,0],
                     [0,1,0,0,0,0]])
    
    # 0:air, 1:wall, 2:goal, 3:key, 4:door
    dimensions = (len(grid), len(grid[0]))
    return (grid, dimensions)

def reset():
    world = create_world()
    grid, _ = world
    return grid

world = create_world()

def step(state, action, world):
    grid, dimensions = world
    width, height = dimensions
    
    prev_row = int(state / width)
    prev_col = state % width
    #print(prev_row)
    #print(prev_col)
    row = prev_row
    col = prev_col
    
    reward = 0
    
    if action == 0:
        row = row - 1
    if action == 1:
        row = row + 1
    if action == 2:
        col = col - 1
    if action == 3:
        col = col + 1
    
    if row < 0:
        row = 0
    if col < 0:
        col = 0
    if row >= width:
        row = width - 1
    if col >= height:
        col = height - 1
    
    done = False
    
    entity = grid[row,col]
    if entity == 0:
        pass
    if entity == 1:
        row = prev_row
        col = prev_col
    if entity == 2:
        reward += 10
        done = True
        #print('done')
    if entity == 3:
        reward += 5
        grid[row,col]=0
        #print('key found')
    if entity == 4:
        keys = 0
        for i in range(width):
            for j in range(height):
                if grid[i,j] == 3:
                    keys+=1
                    #print('there exists a key')
        if keys == 0:
            grid[row,col] = 0
            #print('door unlocked')
        else:
            row = prev_row
            col = prev_col
    world = (grid, dimensions)
    reward = reward - 1
    next_state = row * width + col
    return next_state, reward, done, world

num_episodes = 1000
history = []


grid, dimensions = world
width, height = dimensions
num_states = width * height

action_values = np.zeros((num_states, len(actions)))

epsilon = 0.1
discount_factor = 0.99
alpha = 0.5

def e_greedy_policy(state, values):
    nA = len(values)
    policy = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(values)
    policy[best_action] += (1.0 - epsilon)
    return policy

def greedy_policy(state, values):
    nA = len(values)
    policy = np.zeros(nA, dtype=float)
    best_action = np.argmax(values)
    policy[best_action] += 1
    return policy

average_steps = 0
for i in range(num_episodes):
    # Reset episode
    world = create_world()
    state = int(np.random.uniform() * num_states)
    sequence = []
    done = False
    
    total_reward = 0
    while not done:
        # choose action according to policy
        
        action_probs = e_greedy_policy(state, action_values[state])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # take a step
        next_state, reward, done, world = step(state, action, world)
        
        # update the action values
        next_best_action = np.argmax(action_values[next_state])
        td_target = reward + discount_factor * action_values[next_state, next_best_action]
        td_delta = td_target - action_values[state, action]
        action_values[state, action] += alpha * td_delta
        
        state = next_state
        total_reward += reward
        sequence.append(action)
    
    history.append((sequence, total_reward))
    average_steps += 1/(i+1) * (len(sequence) - average_steps)
    
    if i % int(num_episodes / 20) == 0:
        print('episode ' + str(i))
        print('average steps is ' + str(average_steps))
        print('total reward for this episode: ' + str(total_reward))
        
    
a = action_values.reshape((width, height, 4))
print(np.amax(a, axis = 2).astype(int))

a = np.argmax(a, axis = 2)
print(a)
print(grid)

show_best_action_sequence = True
if show_best_action_sequence:
    state = 0
    sequence = []
    done = False
    steps = 0
    
    while not done and steps < 50:
        # choose action according to policy
        
        action_probs = greedy_policy(state, action_values[state])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # take a step
        next_state, reward, done, world = step(state, action, world)
        
        state = next_state
        sequence.append(action)
        steps+=1
    
    s = []
    for i in sequence:
        s.append(actions[i])
    print(s)
    print('moves: ' + str(steps))

    
            
            