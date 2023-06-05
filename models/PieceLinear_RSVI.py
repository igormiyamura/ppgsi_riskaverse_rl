
import numpy as np, copy, random
from rl_utils.VizTools import VizTools

class PieceLinear_RSVI:
    def __init__(self, env, transition_probabilities, 
                 costs, k, alpha, gamma, num_actions=4, epsilon=0.001, river_flow=None) -> None:
        
        self.viz_tools = VizTools()
        self.env = env
        self._env_name = self.env._env_name
        
        self._river_flow = river_flow
        self._goal_state = self.env._goal_state
        self._num_actions = num_actions
        self._k = k
        self._alpha = alpha
        self._gamma = gamma
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        
        self.V = self.env._build_V0(initial_value=0)
        self.Qi1 = self.env._build_Q0(initial_value=0)
        self.Qi = self.env._build_Q0(initial_value=0)
        self.PI = self.env._build_PI0(initial_value=0)
        
    def __repr__(self):
        if self._env_name == 'RiverProblem':
            self.viz_tools.visualize_V(self, self.V, self.env._grid_size, 4, self.env._goal_state, self._i, 
                                str_title=f'Piecewise Linear VI - RF: {self._river_flow} / k: {self._k}')
            return '> Visualização da Política \n' + \
                f'k: {self._k} \n' + \
                f'alpha: {self._alpha} \n' + \
                f'gamma: {self._gamma} '
        else:
            return ''
            
    def _verify_alpha(self):
        if (self._alpha <= 0) or (self._alpha > (1 + abs(self._k)) ** (-1)): raise Exception(f'Valor de alpha [{self._alpha}] inválido.')
    
    def _verify_k(self):
        if (self._k < -1) or (self._k > 1): raise Exception(f'Valor de k [{self._k}] inválido.')
    
    def _piecewise_linear_transformation(self, value):
        if type(value) == np.ndarray:
            res = []
            for x in value:
                res.append((1 - self._k) * x if x < 0 else (1 + self._k) * x)
            return res
        else:
            x = value
            return (1 - self._k) * x if x < 0 else (1 + self._k) * x
        
    def _reward_function(self, S, action):
        if self._env_name == 'DrivingLicense':
            if S == 'sG': return 0
        
        reward = self._costs[action]
        
        if self._env_name == 'RiverProblem':
            # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
            if S == self._goal_state:
                reward = 0
                
        return reward
    
    def _get_transition(self, S, a, S_Next):
        if self._env_name == 'DrivingLicense': transition_matrix = self._transition_probabilities[S][a]
        elif self._env_name == 'RiverProblem': transition_matrix = self._transition_probabilities[a][S]
        
        return transition_matrix[S_Next]
        
    def _get_V(self):
        V = np.array([v[1] for v in self.V.items()])
        return V
        
    def _get_values_from_dict(self, d):
        V = np.array([v[1] for v in d.items()])
        return V
    
    def define_politica(self, V, goal_state, num_actions):
        PI = {}
        for S in V.keys():
            # if S == goal_state: continue
                
            PI[S] = -1
            action_values = {}
            for action in range(0, num_actions):
                next_state = self._next_state(S, action)
                if next_state != S:
                    action_values[action] = V[next_state]
                    
            PI[S] = min(action_values, key=action_values.get)
        return PI
        
    def run_converge(self):
        done = self.calculate_value()
        return self._i, self.V, self.PI
    
    def calculate_value(self):
        self._i = 0
        
        while True:
            V_prev = copy.deepcopy(self.V)
            Qi1 = copy.deepcopy(self.Qi)
            
            for S in self.V.keys():
                for a in range(self._num_actions):
                    q = 0
                    for S_next in self.V.keys():
                        C = self._reward_function(S, a)
                        q += self._get_transition(S, a, S_next) * self.function_O(Qi1[S_next], Qi1[S][a], C)

                    self.Qi[S][a] = Qi1[S][a] + self._alpha * q
                    
                    # print(f'S: [{S}] / Qi[S]: [{Qi1[S]}] / C: [{C}] / Qp: [{self.Qi[S][a]}] / q: [{q}]')
                    
                self.V[S] = min(self.Qi[S].values())
            
            # print(f'--- V: {self.V} \n --- Qa: {Qi1} \n --- Qp {self.Qi}')
            self._i += 1
            # if self._i == 2:
            #     break
            
            if self.relative_residual(self._get_values_from_dict(self.V), self._get_values_from_dict(V_prev)):
                break
            
        
        # self.PI = self.define_politica(self.V, self._goal_state, self._num_actions)
        # Compute the optimal policy
        Qi = {}
        for S in self.V.keys():
            Qi[S] = {}
            
            for a in range(self._num_actions):
                q = 0
                for S_next in self.V.keys():
                    C = self._reward_function(S, a)
                    q += self._get_transition(S, a, S_next) * self.function_O(Qi1[S_next], Qi1[S][a], C)

                Qi[S][a] = Qi1[S][a] + self._alpha * q
                    
            self.PI[S] = min(Qi[S], key=Qi[S].get)
                
        return True
        
    def function_O(self, Q1, Q2, C):
        Q1min = []
        for a_line in range(self._num_actions):
            Q1min.append(Q1[a_line])
        
        X = self._piecewise_linear_transformation((C + self._gamma * min(Q1min) - Q2))

        return X
    
    def single_function_O(self, Q1, Q2, C):
        X = self._piecewise_linear_transformation((C + self._gamma * Q1 - Q2))

        return X
        
    def relative_residual(self, V1, V2):
        residual = np.abs(np.subtract(V1, V2)/V2)
        # print('Residual: ', max(residual))
        return max(residual) <= self._epsilon
    
    def calculate_value_for_policy(self, Pi, vl_K):
        V = copy.deepcopy(self.env._build_V0())
        i = 0
        
        while True:
            accumulate_cost = 0
            V_prev = copy.deepcopy(V)
            
            for S in V.keys():
                a = Pi[S]
                C = self._reward_function(S, a)
                
                # q = 0
                # for S_next in V.keys():
                #     q += self._transition_probabilities[a][S][S_next] * \
                #         self.single_function_O(V[S_next], V[S], C)
                
                O = self.single_function_O(self._get_values_from_dict(V), V[S], C)
                
                if self._env_name == 'RiverProblem': T = self._get_values_from_dict(self._transition_probabilities[a][S])
                elif self._env_name == 'DrivingLicense': T = self._get_values_from_dict(self._transition_probabilities[S][a])
                # print(f'~ DEBUG - O: [{O}] / T: [{T}]')
                q = sum(T * O)

                V[S] = V[S] + self._alpha * q
                accumulate_cost += V[S]
                # print(f'S: [{S}] / Qi[S]: [{Qi1[S]}] / C: [{C}] / Qp: [{self.Qi[S][a]}] / q: [{q}]')
        
            # print(f'--- V: {V} \n V_PREV: {V_prev}')
            i += 1
            
            if self.relative_residual(self._get_values_from_dict(V), self._get_values_from_dict(V_prev)):
                break
                
        return accumulate_cost