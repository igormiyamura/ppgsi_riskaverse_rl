
import numpy as np
import time

from rl_utils.VizTools import VizTools

class ValueIteration:
    def __init__(self, grid_size, goal_state, transition_probabilities, costs, 
                 num_actions=4, discount_factor=0.95, epsilon=0.001, river_flow=None) -> None:
        self.viz_tools = VizTools()
        
        self._river_flow = river_flow
        self._grid_size = grid_size
        self._rows, self._cols = grid_size[0], grid_size[1]
        self._goal_state = goal_state
        self._num_actions = num_actions
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self.V = self._build_V0()
    
    def __repr__(self):
        self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, 0, 
                               str_title=f'Policy Iteration')
        
        return f'RiverProblem - \n' + \
            f'Discount Factor: {self._discount_factor} \n' + \
            f'Epsilon: {self._epsilon} \n'
    
    def _build_V0(self):
        V0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                V0[(r, c)] = 0
        return V0
        
    def _reward_function(self, S, action):
        reward = self._costs[action]
        
        # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
        if action == 2 and S == (self._goal_state[0], self._goal_state[1] + 1):
            reward = 0
        
        return reward
    
    def _get_transition(self, S, a):
        transition_matrix = self._transition_probabilities[a][S]
        t = np.array([v[1] for v in transition_matrix.items()])
        return t
    
    def _get_V(self):
        V = np.array([v[1] for v in self.V.items()])
        return V
    
    def _next_state(self, state, action):
        x, y = state
        if action == 0:   # up
            x = max(x - 1, 0)
        elif action == 1: # down
            x = min(x + 1, self._rows - 1)
        elif action == 2: # left
            y = max(y - 1, 0)
        elif action == 3: # right
            y = min(y + 1, self._cols - 1)
        return (x, y)
        
    def verify_residual(self, V, V_ANT):
        res = max(np.abs(np.subtract(V, V_ANT)))
        print(f'> Residual: {res}', end='\r')
        return res < 2 * self._epsilon
    
    def step(self):
        V = {}
        for S in self.V.keys():
            bellman = []
            for a in range(0, self._num_actions):
                b = self._reward_function(S, a) + self._discount_factor * \
                    (self._get_transition(S, a) * self._get_V()).sum()
                    
                bellman.append(b)
                
            V[S] = min(bellman)
        
        self.V = V
        return self.V
    
    def run_converge(self):
        start_time = time.time()
        
        qtd_iteracoes = 0
        first = True

        while True:
            first = False

            V_k = [i[1] for i in self.V.items()]

            self.step()
            V_k1 = [i[1] for i in self.V.items()]
            
            qtd_iteracoes += 1
            
            if self.verify_residual(V_k1, V_k):
                break
            
        return qtd_iteracoes, (time.time() - start_time)
    