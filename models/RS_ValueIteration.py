
import numpy as np, random, copy, time
import rl_utils.UtilFunctions as uf

from rl_utils.VizTools import VizTools
from services.ErrorMetrics import ErrorMetrics
from services.ExponentialFunctions import ExponentialFunctions

class RS_ValueIteration:
    def __init__(self, grid_size, goal_state, transition_probabilities, costs, 
                 vl_lambda, num_actions=4, epsilon=0.001, river_flow=None, QUIET=True) -> None:
        self.viz_tools = VizTools()
        
        self._river_flow = river_flow
        self._grid_size = grid_size
        self._rows, self._cols = grid_size[0], grid_size[1]
        self._goal_state = goal_state
        self._num_actions = num_actions
        self._lambda = vl_lambda
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        
        self.PI = self._build_PI0(True, False)
        self.V = self._build_V0()
        self.Qi = self._build_Q0()
        
        self._first_run = True
        self._i = 0
        self.QUIET = QUIET
        
        # Instanciando objetos
        self.EM = ErrorMetrics(self._epsilon)
        self.EF = ExponentialFunctions()
    
    def __repr__(self):
        self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                               str_title=f'Exponential Utility Function - RSMDP - Lambda {self._lambda}')
        
        return f'RiverProblem - \n' + \
            f'Lambda: {self._lambda} \n' + \
            f'Epsilon: {self._epsilon} \n'
    
    def _build_PI0(self, random=True, proper=False):
        PI0 = {}
        if proper:
            # Preenche todos os blocos de terra
            for c in range(0, self._cols): 
                PI0[(0, c)] = 3
                PI0[(self._rows - 1, c)] = 2
            for r in range(0, self._rows - 1):
                PI0[(r, self._cols - 1)] = 1
            # Preenche blocos waterfall
            for r in range(1, self._rows-1):
                PI0[(r, 0)] = self._get_random_action() if random else 0
            # Preenche todos os blocos de rio
            for r in range(1, self._rows-1):
                for c in range(1, self._cols-1):
                    PI0[(r, c)] = self._get_random_action() if random else 0
        else:
            for r in range(0, self._rows):
                for c in range(0, self._cols):
                    PI0[(r, c)] = self._get_random_action() if random else 0
        
        return PI0
    
    def _build_V0(self, zero=False):
        V0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                # Preenche o V0 com o negativo do sinal do Lambda
                V0[(r, c)] = np.sign(self._lambda)
        return V0
    
    def _build_Q0(self):
        Q0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                Q0[(r, c)] = {}
                for a in range(self._num_actions):
                    Q0[(r, c)][a] = 0
        return Q0
        
    def _cost_function(self, S, action):
        reward = self._costs[action]
        
        # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
        if S == (self._goal_state[0], self._goal_state[1]):
            reward = 0
        
        return reward
    
    def run_converge(self):
        start_time = time.time()
        self.calculate_value()
        
        return self._i, (time.time() - start_time)
                
    def calculate_value(self):
        while True:
            new_V = {}
            
            for S in self.V.keys():
                b = []
                
                for a in range(self._num_actions):
                    V = uf._get_values_from_dict(self.V)
                    T = uf._get_values_from_dict(self._transition_probabilities[a][S])
                    
                    TV = T * V
                    C = self._cost_function(S, a)
                    
                    bellman = self.EF._utility(C, self._lambda) * sum(TV)
                    b.append(bellman)
                    
                    if not self.QUIET: print(f'b: {b} | TV: {TV} | C: {C} | Lambda: {self._lambda}')
                
                if S == self._goal_state:
                    new_V[S] = np.sign(self._lambda)
                else:
                    new_V[S] = min(b)
            self._i += 1
            
            if self.EM.relative_residual(uf._get_values_from_dict(self.V), uf._get_values_from_dict(new_V)):
                break

            self.V = copy.deepcopy(new_V)
            
        # Compute the optimal policy
        for S in self.V.keys():
            for a in range(self._num_actions):
                q = uf._get_values_from_dict(self._transition_probabilities[a][S]) * uf._get_values_from_dict(self.V)
                C = self._cost_function(S, a)
                self.Qi[S][a] = np.exp(self._lambda * C) * sum(q)
                    
            self.PI[S] = min(self.Qi[S], key=self.Qi[S].get)
            
    def calculate_value_for_policy(self, Pi, vl_lambda):
        i = 0
        
        while True:
            new_V = {}
            for S in Pi.keys():
                a = Pi[S]
                
                q = uf._get_values_from_dict(self._transition_probabilities[a][S]) * uf._get_values_from_dict(self.V)
                C = self._cost_function(S, a)
                
                if S == self._goal_state:
                    new_V[S] = np.sign(self._lambda)
                else:
                    new_V[S] = np.exp(self._lambda * C) * sum(q)
            
            # print(f'Exp: {np.exp(vl_lambda * C)} / q: {sum(q)} ')
            i += 1
        
            if self.EM.relative_residual(uf._get_values_from_dict(self.V), uf._get_values_from_dict(new_V)):
                break
            
            self.V = copy.deepcopy(new_V)
            
        accumulate_cost = sum(uf._get_values_from_dict(self.V))            
        return accumulate_cost