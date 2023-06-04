
import numpy as np, random, copy, time
import rl_utils.UtilFunctions as uf

from rl_utils.VizTools import VizTools
from services.ErrorMetrics import ErrorMetrics
from services.ExponentialFunctions import ExponentialFunctions

class ExponentialUtility_RSVI:
    def __init__(self, env, transition_probabilities, costs, 
                 vl_lambda, num_actions=4, epsilon=0.001, river_flow=None, QUIET=True) -> None:
        self.viz_tools = VizTools()
        self.env = env
        
        self._env_name = self.env._env_name
        self._river_flow = river_flow
        self._num_actions = num_actions
        self._lambda = vl_lambda
        self._threshold = 1000
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        self._goal_state = env._goal_state
        
        self.PI = self.env._build_PI0(initial_value=0)
        self.V = self.env._build_V0(initial_value=self._define_initial_value_V0())
        self.Qi = self.env._build_Q0(initial_value=0)
        
        self._first_run = True
        self._i = 0
        self.QUIET = QUIET
        
        # Instanciando objetos
        self.EM = ErrorMetrics(self._epsilon)
        self.EF = ExponentialFunctions()
    
    def __repr__(self):
        if self._env_name == 'RiverProblem':
            self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                                str_title=f'Exponential Utility Function - RSMDP - Lambda {self._lambda}')
            
            return f'RiverProblem - \n' + \
                f'Lambda: {self._lambda} \n' + \
                f'Epsilon: {self._epsilon} \n'
        else:
            return ''
    
    def _define_initial_value_V0(self):
        if self._env_name == 'DrivingLicense': return np.sign(self._lambda)
        elif self._env_name == 'RiverProblem': return np.sign(self._lambda)
        else: return 0
    
    def _get_transition(self, S, a):
        if self._env_name == 'DrivingLicense': transition_matrix = self._transition_probabilities[S][a]
        elif self._env_name == 'RiverProblem': transition_matrix = self._transition_probabilities[a][S]
        t = uf._get_values_from_dict(transition_matrix)
        return t    
        
    def _cost_function(self, S, action):
        if self._env_name == 'DrivingLicense':
            if S == 'sG': return 0
        
        reward = self._costs[action]
        
        if self._env_name == 'RiverProblem':
            # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
            if S == self._goal_state:
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
                    T = self._get_transition(S, a) # uf._get_values_from_dict(self._transition_probabilities[a][S])
                    
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
                q = self._get_transition(S, a) * uf._get_values_from_dict(self.V)
                C = self._cost_function(S, a)
                self.Qi[S][a] = np.exp(self._lambda * C) * sum(q)
                    
            self.PI[S] = min(self.Qi[S], key=self.Qi[S].get)
            
    def calculate_value_for_policy(self, Pi, vl_lambda):
        i = 0
        
        while True and i < self._threshold:
            new_V = {}
            for S in Pi.keys():
                a = Pi[S]
                
                q = self._get_transition(S, a) * uf._get_values_from_dict(self.V)
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