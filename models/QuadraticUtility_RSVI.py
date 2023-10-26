
import numpy as np, random, copy, time
import rl_utils.UtilFunctions as uf

from models.Neutral_VI import Neutral_VI
from rl_utils.VizTools import VizTools
from services.ErrorMetrics import ErrorMetrics

class QuadraticUtility_RSVI:
    def __init__(self, env, transition_probabilities, costs, 
                 vl_slope1, vl_slope2, num_actions=4, epsilon=0.001, river_flow=None, discount_factor=0.99, QUIET=True) -> None:
        self.viz_tools = VizTools()
        self.env = env
        
        self._env_name = self.env._env_name
        self._river_flow = river_flow
        self._num_actions = num_actions
        self._slope1 = vl_slope1
        self._slope2 = vl_slope2
        self._threshold = 1000
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self._goal_state = env._goal_state
        
        self.PI = self.env._build_PI0(initial_value=0)
        self.V = self.env._build_V0(initial_value=self._define_initial_value_V0())
        self.Qi = self.env._build_Q0(initial_value=0)
        
        self._first_run = True
        self._i = 0
        self.QUIET = QUIET
        
        # Guardando informação de convergencia
        self._inicia_historico_calculo()
        
        # Instanciando objetos
        self.EM = ErrorMetrics(self._epsilon)
    
    def __repr__(self):
        if self._env_name == 'RiverProblem':
            self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                                str_title=f'Exponential Utility Function - RSMDP - Slope1 {self._slope1} | Slope2 {self._slope2}')
            
            return f'RiverProblem - \n' + \
                f'Slope1: {self._slope1} \n' + \
                f'Slope2: {self._slope2} \n' + \
                f'Epsilon: {self._epsilon} \n'
        else:
            return ''
    
    def _inicia_historico_calculo(self):
        self._hist_custo = {}
        self._hist_probabilty = {}
        self._hist_s0 = {}
        self._hist_G = {}
        self._policy_value = {}
        
        for S in self.V.keys():
            self._hist_custo[S] = {}
            self._hist_probabilty[S] = {}
            self._hist_s0[S] = {}
            self._hist_G[S] = {}
            self._policy_value[S] = {}
            for a in range(self._num_actions):
                self._hist_custo[S][a] = []
                self._hist_probabilty[S][a] = []
                self._hist_s0[S][a] = []
                self._hist_G[S][a] = []
                self._policy_value[S][a] = []
                
        return True
    
    def _define_initial_value_V0(self):
        if self._env_name == 'DrivingLicense': return 0
        elif self._env_name == 'RiverProblem': return 0
        else: return 0
    
    def _get_transition(self, S, a):
        if self._env_name == 'DrivingLicense': transition_matrix = self._transition_probabilities[S][a]
        elif self._env_name == 'RiverProblem': transition_matrix = self._transition_probabilities[a][S]
        else: transition_matrix = self._transition_probabilities[S][a]
        
        t = uf._get_values_from_dict(transition_matrix)
        return t    
        
    def _cost_function(self, S, action):
        if self._env_name == 'DrivingLicense' or self._env_name == 'SimpleMDP':
            if S == 'sG': return 0
        
        reward = self._costs[action]
        
        if self._env_name == 'RiverProblem':
            # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
            if S == self._goal_state:
                reward = 0
                
        return reward
    
    def run_converge(self):
        V, PI = self.calculate_value()
        
        return self._i, V, PI
                
    def calculate_value(self):
        while True and self._i < self._threshold:
            new_V = {}
            
            for S in self.V.keys():
                b = []
                
                for a in range(self._num_actions):
                    V = uf._get_values_from_dict(self.V)
                    T = self._get_transition(S, a)
                    
                    TV = T * V
                    C = self._cost_function(S, a)
                    
                    bellman = (self._slope2 * C**2 + self._slope1 * C) + sum(TV)
                    b.append(bellman)
                    
                    if not self.QUIET: print(f'b: {b} | TV: {TV} | C: {C} | Slope: {self._slope}')

                    self._hist_custo[S][a].append((self._slope2 * C**2 + self._slope1 * C))
                    self._hist_probabilty[S][a].append(sum(TV))
                    self._hist_s0[S][a].append((self._slope2 * C**2 + self._slope1 * C) + T[0] * V[0])
                    self._hist_G[S][a].append((self._slope2 * C**2 + self._slope1 * C) + T[1] * V[1])
                    self._policy_value[S][a].append(bellman)
                    
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
                self.Qi[S][a] = (self._slope2 * C**2 + self._slope1 * C) + sum(q)
                    
            self.PI[S] = min(self.Qi[S], key=self.Qi[S].get)
            
        return self.V, self.PI
            
    def calculate_value_for_policy(self, Pi, vl_slope):
        i = 0
        
        while True and i < self._threshold:
            new_V = {}
            for S in Pi.keys():
                a = Pi[S]
                
                q = self._get_transition(S, a) * uf._get_values_from_dict(self.V)
                C = self._cost_function(S, a)
                
                new_V[S] =  (self._slope2 * C**2 + self._slope1 * C) + sum(q)
            
            # print(f'Exp: {np.exp(vl_slope * C)} / q: {sum(q)} ')
            i += 1
        
            if self.EM.relative_residual(uf._get_values_from_dict(self.V), uf._get_values_from_dict(new_V)):
                break
            
            self.V = copy.deepcopy(new_V)
            
        accumulate_cost = sum(uf._get_values_from_dict(self.V))            
        return accumulate_cost