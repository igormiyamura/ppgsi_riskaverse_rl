import numpy as np

# Define the reward function
def reward_function(state, action, next_state):
    reward = -1
    if next_state[0] == goal[0] and next_state[1] == goal[1]:
        reward = 10
    return reward

# Define the transition function
def transition_function(state, action):
    next_state = state + actions[action]
    if next_state[0] < 0 or next_state[0] >= rows or next_state[1] < 0 or next_state[1] >= cols:
        next_state = state
    return next_state

# Define the CVaR function
def CVaR(values, alpha):
    values = np.sort(values)
    n = len(values)
    values = np.clip(values, -1e10, 1e10) # Clamp the values to a maximum and minimum value
    cvar = (1/alpha) * (np.sum(values[:int(alpha*n)]) + (alpha*n-int(alpha*n))*values[int(alpha*n)])
    return cvar


# Define the probabilistic planning algorithm
def probabilistic_planning(alpha, num_iterations):
    V = np.zeros((rows,cols))
    for iteration in range(num_iterations):
        for row in range(rows):
            for col in range(cols):
                values = []
                for action in range(num_actions):
                    next_state = transition_function([row,col], action)
                    reward = reward_function([row,col], action, next_state)
                    values.append(reward + discount_factor*V[next_state[0], next_state[1]])
                V[row,col] = CVaR(values, alpha_cvar)
    return V

# Define the parameters
alpha = 0.1
num_iterations = 1000
alpha_cvar = 0.95
rows = 5
cols = 5
num_actions = 4
actions = [[-1,0], [1,0], [0,-1], [0,1]]
discount_factor = 0.9
goal = [4,4]

# Run the probabilistic planning algorithm with CVaR
V = probabilistic_planning(alpha, num_iterations)
