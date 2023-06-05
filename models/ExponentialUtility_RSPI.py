
import numpy as np, random, copy, math, time
import rl_utils.UtilFunctions as uf

from models.Neutral_PI import Neutral_PI
from rl_utils.VizTools import VizTools
from services.ErrorMetrics import ErrorMetrics
from services.ExponentialFunctions import ExponentialFunctions

class ExponentialUtility_RSPI:
    def __init__(self, env, transition_probabilities, costs, 
                 vl_lambda, num_actions=4, epsilon=0.001, river_flow=None, certainty_equivalent=False, 
                 explogsum=False, threshold=1000, discount_factor=0.99, QUIET=True) -> None:
        self.viz_tools = VizTools()
        self.env = env
        
        self._env_name = self.env._env_name
        self._goal_state = self.env._goal_state
        self._river_flow = river_flow
        self._num_actions = num_actions
        self._lambda = vl_lambda
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self._threshold = threshold
        
        self._certainty_equivalent = certainty_equivalent
        self._explogsum = explogsum
        
        self.V = self.env._build_V0(initial_value=self._define_initial_value_V0())
        self.PI = self.env._build_PI0(initial_value=0)
        
        self._first_run = True
        self._i = 0
        self.QUIET = QUIET
        
        # Instanciando objetos
        self.EM = ErrorMetrics(self._epsilon)
        self.EF = ExponentialFunctions()
        
    def __repr__(self):
        if self._env_name == 'RiverProblem':
            self.viz_tools.visualize_V(self, self.V, self.env._grid_size, 4, self._goal_state, self._i, 
                                str_title=f'Exponential Utility Function - RSMDP - Lambda {self._lambda}')
            
            return f'RiverProblem - \n' + \
                f'Lambda: {self._lambda} \n' + \
                f'Epsilon: {self._epsilon} \n'
        else:
            return None
    
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
        if self._lambda == 0:
            self.N_PI = Neutral_PI(self.env, self._transition_probabilities, self._costs, 
                 num_actions=self._num_actions, discount_factor=self._discount_factor, epsilon=self._epsilon, river_flow=self._river_flow)
            i, V, PI = self.N_PI.run_converge()
            
            self._i = i
            self.V = V
            self.PI = PI
        else:
            while(self._first_run or (self.PI != self.PI_ANT) and self._i < self._threshold):
                V, PI = self.step()
                
                self._first_run = False
                self._i += 1
            
        return self._i, V, PI
        
    def step(self):
        self.policy_evaluation()
        self.policy_improvement()
        
        return self.V, self.PI
    
    def policy_evaluation(self):
        i = 0
        
        while True and i < self._threshold:
            NEW_V = {}
            for S in self.V.keys():
                pi_a = self.PI[S]
                
                if S == self._goal_state:
                    bellman = np.sign(self._lambda)
                else:
                    C = self._cost_function(S, pi_a)
                    l = self._lambda
                    T = self._get_transition(S, pi_a) # uf._get_values_from_dict(self._transition_probabilities[pi_a][S])
                    V = uf._get_values_from_dict(self.V)
                    
                    if not self._explogsum:
                        TV = T * V
                        bellman = self.EF._utility(C, l) * sum(TV)
                    else:
                        kai = np.log(T) + l * V
                        KAI = max(kai)
                        bellman = C + KAI / l + (1 / l) * np.log(sum(np.exp(kai - KAI)))
                        
                    if not self.QUIET: print(f'({S}) Lambda: {self._lambda} / C: {C} / exp(lC): {np.exp(l * C)} / Bellman: {bellman}')
                
                NEW_V[S] = bellman
            
            if not self.QUIET: print(f'NEW_V: {NEW_V}')
            if self.EM.relative_residual(uf._get_values_from_dict(self.V), 
                                      uf._get_values_from_dict(NEW_V)):
                break
            
            self.V = NEW_V
            i += 1
            if not self.QUIET: print(f'i: {i}', end='\r')
        
        # print(f'V: {self.V}')
        return self.V
    
    def policy_improvement(self):
        if not self.QUIET: print('>>>>>>>>>>> POLICY IMPROVEMENT')
        self.PI_ANT = copy.deepcopy(self.PI)
        
        pi_improved = {}
        for S in self.V.keys():
            bellman = {}
            # improve the current policy by doing  the following update for every s ∈ S
            for a in range(0, self._num_actions):
                TV = self._get_transition(S, a) * uf._get_values_from_dict(self.V)
                
                bellman[a] = np.exp(self._lambda * self._cost_function(S, a)) * sum(TV)
                
            pi_improved[S] = min(bellman, key=bellman.get)
        
        self.PI = copy.deepcopy(pi_improved)
        return self.PI
    
    def calculate_value_for_policy(self, Pi, vl_lambda):
        i = 0
        
        while True and i < self._threshold:
            NEW_V = {}
            for S in self.V.keys():
                pi_a = Pi[S]
                
                if S == self._goal_state:
                    bellman = np.sign(vl_lambda)
                else:
                    C = self._cost_function(S, pi_a)
                    l = vl_lambda
                    
                    T = self._get_transition(S, pi_a) # uf._get_values_from_dict(self._transition_probabilities[pi_a][S])
                    V = uf._get_values_from_dict(self.V)
                    
                    if not self._explogsum:
                        TV = T * V
                        bellman = self.EF._utility(C, l) * sum(TV)
                    else:
                        kai = np.log(T) + l * V
                        KAI = max(kai)
                        bellman = C + KAI / l + (1 / l) * np.log(sum(np.exp(kai - KAI)))
                    
                    if not self.QUIET: print(f'({S}) Lambda: {vl_lambda} / C: {C} / exp(lC): {np.exp(l * C)} / TV: {sum(TV)} / Bellman: {bellman}')
                        
                NEW_V[S] = bellman
            
            if not self.QUIET: print(f'V: {self.V} || V_NEW: {V}')
            if self.EM.relative_residual(uf._get_values_from_dict(self.V), 
                                      uf._get_values_from_dict(NEW_V)):
                break
            
            self.V = NEW_V
            i += 1
            if not self.QUIET: print(f'i: {i}', end='\r')
        
        accumulate_cost = sum(uf._get_values_from_dict(self.V))
        return accumulate_cost