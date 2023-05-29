
import numpy as np, random, copy, math, time
import rl_utils.UtilFunctions as uf

from rl_utils.VizTools import VizTools
from services.ErrorMetrics import ErrorMetrics
from services.ExponentialFunctions import ExponentialFunctions

class ExponentialUtility_RSPI:
    def __init__(self, env, transition_probabilities, costs, 
                 vl_lambda, num_actions=4, epsilon=0.001, river_flow=None, certainty_equivalent=False, 
                 explogsum=False, threshold=1000, QUIET=True) -> None:
        self.viz_tools = VizTools()
        self._env = env
        
        self._river_flow = river_flow
        self._grid_size = env._grid_size
        self._rows, self._cols = env._grid_size[0], env._grid_size[1]
        self._goal_state = env._goal_state
        self._num_actions = num_actions
        self._lambda = vl_lambda
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        self._threshold = threshold
        
        self._certainty_equivalent = certainty_equivalent
        self._explogsum = explogsum
        
        self.V = self._build_V0()
        self.PI = self._build_PI0(True, True)
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
                PI0[(r, 0)] = self._env._get_random_action(self._num_actions) if random else 0
            # Preenche todos os blocos de rio
            for r in range(1, self._rows-1):
                for c in range(1, self._cols-1):
                    PI0[(r, c)] = self._env._get_random_action(self._num_actions) if random else 0
        else:
            for r in range(0, self._rows):
                for c in range(0, self._cols):
                    PI0[(r, c)] = self._env._get_random_action(self._num_actions) if random else 0
        
        return PI0
    
    def _build_V0(self):
        V0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                # Preenche o V0 com o negativo do sinal do Lambda
                V0[(r, c)] = np.sign(self._lambda)
        return V0
    
    def _cost_function(self, S, action):
        reward = self._costs[action]
        
        # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
        if S == (self._goal_state[0], self._goal_state[1]):
            reward = 0
        
        return reward
    
    def run_converge(self):
        start_time = time.time()

        while(self._first_run or (self.PI != self.PI_ANT)):
            print(f'[L: {self._lambda}] Iteração: {self._i}...', end='\r')
            self.step()
            
            self._first_run = False
            self._i += 1
            
        return self._i, (time.time() - start_time)
        
    def step(self):
        self.policy_evaluation()
        self.policy_improvement()
    
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
                    T = uf._get_values_from_dict(self._transition_probabilities[pi_a][S])
                    V = uf._get_values_from_dict(self.V)
                    
                    if not self._explogsum:
                        TV = T * V
                        bellman = self.EF._utility(C, l) * sum(TV)
                    else:
                        kai = np.log(T) + l * V
                        KAI = max(kai)
                        bellman = C + KAI / l + (1 / l) * np.log(sum(np.exp(kai - KAI)))
                        
                    if not self.QUIET: print(f'({S}) Lambda: {self._lambda} / C: {C} / exp(lC): {np.exp(l * C)} / TV: {TV} / Bellman: {bellman}')
                
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
                TV = uf._get_values_from_dict(self._transition_probabilities[a][S]) * uf._get_values_from_dict(self.V)
                
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
                    
                    T = uf._get_values_from_dict(self._transition_probabilities[pi_a][S])
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