import numpy as np
import time
import json


def generate_board():
    global board
    board = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0]])

def reset():
    generate_board()
    


def transformed_boards(board):
    boards = []
    boards.append(board.copy())
    boards.append(np.rot90(board,k=1))
    boards.append(np.rot90(board,k=2))
    boards.append(np.rot90(board,k=3))
    flip = np.flip(board, 0)
    boards.append(flip)
    boards.append(np.rot90(flip,k=1))
    boards.append(np.rot90(flip,k=2))
    boards.append(np.rot90(flip,k=3))
    return boards

def available_actions(board):
    actions = []
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r,c] == 0:
                actions.append(r * 3 + c)
    return actions

def init():
    generate_board()
    
    global states
    states = []
    
    global q_table
    q_table = dict()
    
    global lookup
    lookup = dict()
    
    global mappings
    mappings = []
    t = transformed_boards(np.arange(9).reshape((3,3)))
    for e in t:
        mappings.append(dict((a,b) for b,a in enumerate(e.reshape((9,)))))
    # one mapping is, original_action : transform_action
    

def update_tables(board):
    global q_table
    global mappings
    global lookup
    
    # for cheaper computation
    for i,e in enumerate(transformed_boards(board)):
        h = hash(e.tobytes())
        if h not in lookup.keys():
            lookup.update({h:(hash(board.tobytes()), i)})
            
    actions = available_actions(board)
    #action_values = {k:np.random.rand()*0.1 for k in actions}
    action_value_pairs = {k:0 for k in actions}
    q_table.update({hash(board.tobytes()):action_value_pairs})
    
def q_function(state):
    global lookup
    global q_table
    
    board = state
    h = hash(board.tobytes())
    if hash(board.tobytes()) not in lookup.keys():
        update_tables(board)
    
    original_board_hash, m = lookup[h]
    global mappings
    mapping = mappings[m]
    
    action_value_pairs = q_table[original_board_hash]
    #but we need to convert it to the transformation it is in
    action_values = dict()
    #print(mapping)
    #print(action_value_pairs)
    for key in action_value_pairs:
        value = action_value_pairs[key]
        new_key = mapping[key]
        action_values.update({new_key:value})
    
    #print(action_values)
    return action_values

def q_update(state, action, update):
    global lookup
    global q_table
    
    board = state
    h = hash(board.tobytes())
    if hash(board.tobytes()) not in lookup.keys():
        update_tables(board)
    
    original_board_hash, m = lookup[h]
    global mappings
    mapping = mappings[m]
    
    action_value_pairs = q_table[original_board_hash]
    
    for key in action_value_pairs:
        value = action_value_pairs[key]
        new_key = mapping[key]
        if new_key == action:
            value += update
            q_table[original_board_hash].update({key:value})
    return
    
    
def save_tables():
    global q_table
    try:
        with open('q_table.json', 'w') as fp:
            json.dump(q_table, fp, sort_keys=False, indent=4)
    except:
        print('failed saving q_table.json')
    
    global lookup
    try:
        with open('lookup.json', 'w') as fp2:
            json.dump(lookup, fp2, indent=4)
    except:
        print('failed saving lookup.json')

def load_tables():
    global q_table
    try:
        with open('q_table.json', 'r') as fp:
            q_table = json.load(fp)
    except:
        print('failed loading q_table.json')
    
    global lookup
    try:
        with open('lookup.json', 'r') as fp2:
            lookup = json.load(fp2)
    except:
        print('failed loading lookup.json')
    
def make_move(player, action):
    global board
    r = int(action / 3)
    c = int(action % 3)
    if board[r,c] == 0:
        if player == 'x':
            board[r,c] = 1
        if player == 'o':
            board[r,c] = 2
    else:
        print('Warning: illegal move atempted')
        print('row: ' + str(r + 1) + ' col: ' + str(c + 1))
        print_board()

def check_board():
    global board
    x_win = False
    o_win = False
    tie = False
    for r in range(len(board)):
        if board[r,0] == board[r,1] and board[r,1] == board[r,2]:
            if board[r,0] == 0:
                continue
            elif board[r,0] == 1:
                x_win = True
                return x_win, o_win, tie
            elif board[r,0] == 2:
                o_win = True
                return x_win, o_win, tie
    
    for c in range(len(board[0])):
        if board[0,c] == board[1,c] and board[1,c] == board[2,c]:
            if board[0,c] == 0:
                continue
            elif board[0,c] == 1:
                x_win = True
                return x_win, o_win, tie
            elif board[0,c] == 2:
                o_win = True
                return x_win, o_win, tie
    
    if board[0,0] == board[1,1] and board[1,1] == board[2,2]:
        if board[0,0] == 0:
            pass
        elif board[0,0] == 1:
            x_win = True
            return x_win, o_win, tie
        elif board[0,0] == 2:
            o_win = True
            return x_win, o_win, tie
            
    if board[0,2] == board[1,1] and board[1,1] == board[2,0]:
        if board[0,2] == 0:
            pass
        elif board[0,2] == 1:
            x_win = True
            return x_win, o_win, tie
        elif board[0,2] == 2:
            o_win = True
            return x_win, o_win, tie

    
    tie = True
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r,c] == 0:
                tie = False
    return x_win, o_win, tie

def is_done(x_win, o_win, tie):
    return x_win or o_win or tie

def get_reward(x_win, o_win):
    win_reward = 10
    lose_reward = -10
    if x_win:
        return win_reward
    if o_win:
        return lose_reward
    return 0

def step(player, action):
    
    make_move(player, action)
    
    x_win, o_win, tie = check_board()
    global wins, losses, ties
    if x_win:
        wins += 1
    if o_win:
        losses += 1
    if tie:
        ties+=1
    
    
    reward = get_reward(x_win, o_win)
    
    done = is_done(x_win, o_win, tie)
    
    return reward, done


def e_greedy_policy(player, values, epsilon = 0.1):
    nA = len(values)
    policy = np.ones(nA, dtype=float) * epsilon / nA
    if player == 'x':
        best_action = np.argmax(values)
    else:
        best_action = np.argmin(values)
    policy[best_action] += (1.0 - epsilon)
    return policy

def greedy_policy(player, values):
    nA = len(values)
    policy = np.zeros(nA, dtype=float)
    if player == 'x':
        best_action = np.argmax(values)
    else:
        best_action = np.argmin(values)
    policy[best_action] += 1
    return policy

def random_policy(values):
    nA = len(values)
    policy = np.ones(nA, dtype=float) / nA
    return policy

def print_board():
    global board
    b = ''
    c = 0
    d = 0
    for row in board:
        b+=' '
        for i in row:
            if i == 0:
                b += ' '
            elif i == 1:
                b += 'X'
            elif i == 2:
                b += 'O'
            if c % 3 != 2:
                b += ' | '
            c+=1
        b+=' '
        if d != 2:
            b+='\n___|___|___\n   |   |   \n'
        d+=1
        
    b+='\n\n'
    print(b)

def train(num_episodes = 1000, train = True, greedy = False, random = False, verbose = False, discount_factor = 0.90, alpha = 0.5, epsilon = 0.1):
    start_time = time.clock()
    beginning = start_time
    time_since_last_save = beginning
    for i in range(num_episodes):
        if num_episodes >= 1000000: 
            epsilon = 10**(-(np.sin(np.pi * 2 * i / (num_episodes / 200)) + 1))
            alpha = np.sin(np.pi * 2 * i / (num_episodes / 300))/2.3 + 0.5
        
        # Define global variables we will use
        global board
        global q_table
        # Reset episode
        reset()
        player = 'x'
        rival = 'o'
        # Update the starting state
        state = board.copy()      
        
        done = False
        while not done:
            if verbose:
                print_board()
            
            # choose action according to policy
            q = q_function(state)
            values = list(q.values())
            if not greedy:
                action_probs = e_greedy_policy(player, values, epsilon = epsilon)
            else:
                if not random:
                    action_probs = greedy_policy(player, values)
                else:
                    action_probs = random_policy(values)
            index = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            actions_available = list(q.keys())
            action = actions_available[index]
            
            # take a step
            reward, done = step(player, action)
            next_state = board.copy()
            
            if verbose:
                print(np.sum(values)/len(values))
            
            if train:
                # update the action values
                q_next = q_function(next_state)
                actions_available = list(q_next.keys())
                if len(actions_available) != 0:
                    values = list(q_next.values())
                    index = np.argmax(values)
                    next_best_action = actions_available[index]
                    
                    td_target = reward + discount_factor * q_next[next_best_action]
                else:
                    td_target = reward
                    #print('no moves left')
                td_delta = td_target - q[action]
                update = alpha * td_delta
                q_update(state, action, update)
            
            state = next_state
            
            temp = player
            player = rival
            rival = temp
        
        if verbose:
            print_board()
            x_win, o_win, tie = check_board()
            if x_win:
                print('Player X wins')
            if o_win:
                print('Player O wins')
            if tie:
                print('Nobody wins')
        if num_episodes >= 10000 and i % int(num_episodes / 100) == 0:
            percent = int(i / int(num_episodes / 100)) + 1
            print(str(percent) + '% complete...')
        
        if time.clock() - time_since_last_save > 30 * (np.log10(num_episodes)) / 10:
            print('Saving...')
            save_tables()
            time_left = (time.clock() - beginning) / (i + 1) * (num_episodes - i)
            minutes = int(time_left / 60)
            seconds = time_left % 60
            hours = int(minutes / 60)
            minutes = minutes % 60
            print('Estimated time remaining: ' + str(int(hours)) + ' hours, ' + str(int(minutes)) + ' minutes, ' + str(int(seconds)) + ' seconds')
            print('epsilon: ' + str(epsilon) + ' alpha: ' + str(alpha))
            time_since_last_save = time.clock()
wins, losses, ties = 0, 0, 0
reset()
init()

load_tables()

print('training...')
train(num_episodes=1000000, random = True, alpha=0.7)

save_tables()
print('Progress saved.')

if True:
    train(num_episodes=1, greedy = True, train = False, verbose = True)

    print(wins)
    print(losses)
    print(ties)
      
