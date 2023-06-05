
import numpy as np
import time

from rl_utils.VizTools import VizTools

class Neutral_VI:
    def __init__(self, env, transition_probabilities, costs, 
                 num_actions=4, discount_factor=0.95, epsilon=0.001, river_flow=None) -> None:
        self.env = env
        self.viz_tools = VizTools()
        
        self._env_name = self.env._env_name
                
        self._river_flow = river_flow
        self._num_actions = num_actions
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self.V = env._build_V0(initial_value=0)
        self.PI = env._build_PI0(initial_value=-1)
    
    def __repr__(self):
        if self._env_name == 'RiverProblem':
            self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, 0, 
                                str_title=f'Policy Iteration')
            
            return f'{self.env._env_name} - \n' + \
                f'Discount Factor: {self._discount_factor} \n' + \
                f'Epsilon: {self._epsilon} \n'
        else:
            return None
        
    def _reward_function(self, S, action):
        if self._env_name == 'DrivingLicense':
            if S == 'sG': return 0
        
        reward = self._costs[action]
        
        if self._env_name == 'RiverProblem':
            # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
            if S == self._goal_state:
                reward = 0
                
        return reward
    
    def _get_transition(self, S, a):
        if self._env_name == 'DrivingLicense': transition_matrix = self._transition_probabilities[S][a]
        elif self._env_name == 'RiverProblem': transition_matrix = self._transition_probabilities[a][S]
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
        # print(f'> Residual: {res}', end='\r')
        return res < 2 * self._epsilon
    
    def step(self):
        V = {}
        PI = {}
        
        for S in self.V.keys():
            bellman = {}
            for a in range(0, self._num_actions):
                b = self._reward_function(S, a) + self._discount_factor * \
                    (self._get_transition(S, a) * self._get_V()).sum()
                    
                bellman[a] = b
                
            PI[S] = min(bellman, key=bellman.get)
            V[S] = min([v[1] for v in bellman.items()])
        
        self.V = V
        self.PI = PI
        return self.V, self.PI
    
    def run_converge(self):
        start_time = time.time()
        
        qtd_iteracoes = 0
        first = True

        while True:
            first = False

            V_k = [i[1] for i in self.V.items()]

            V, PI = self.step()
            V_k1 = [i[1] for i in self.V.items()]
            
            qtd_iteracoes += 1
            
            if self.verify_residual(V_k1, V_k):
                break
            
        return qtd_iteracoes, V, PI