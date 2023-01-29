import gym
from gym import spaces
import numpy as np

class StochasticGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.grid_size = (5, 5)
        self.goal_state = (4, 4)
        self.action_space = spaces.Discrete(4) # up, down, left, right
        self.observation_space = spaces.Discrete(np.prod(self.grid_size))
        self.transition_probabilities = {
            0: {0: 0.8, 1: 0.1, 2: 0.1, 3: 0}, # up
            1: {0: 0.1, 1: 0.8, 2: 0, 3: 0.1}, # down
            2: {0: 0, 1: 0.1, 2: 0.8, 3: 0.1}, # left
            3: {0: 0.1, 1: 0, 2: 0.1, 3: 0.8}  # right
        }
        self.costs = {0: -1, 1: -1, 2: -1, 3: -1}
        self.reset()
        
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        assert self.action_space.contains(action)
        prob = np.random.uniform(0, 1)
        next_state = self.state
        for i in range(4):
            if prob <= sum(self.transition_probabilities[action].values()[:i+1]):
                next_state = self._move(next_state, i)
                break
                
        reward = self.costs[action]
        done = next_state == self.goal_state
        return next_state, reward, done, {}
    
    def _move(self, state, action):
        x, y = state
        if action == 0:   # up
            x = max(x - 1, 0)
        elif action == 1: # down
            x = min(x + 1, self.grid_size[0] - 1)
        elif action == 2: # left
            y = max(y - 1, 0)
        elif action == 3: # right
            y = min(y + 1, self.grid_size[1] - 1)
        return (x, y)
