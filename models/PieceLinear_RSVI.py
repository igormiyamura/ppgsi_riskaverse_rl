
import numpy as np, copy
from rl_utils.VizTools import VizTools

class PieceLinear_RSVI:
    def __init__(self, grid_size, goal_state, transition_probabilities, 
                 costs, k, alpha, gamma, num_actions=4, epsilon=0.001, river_flow=None) -> None:
        
        self.viz_tools = VizTools()
        
        self._river_flow = river_flow
        self._grid_size = grid_size
        self._rows, self._cols = grid_size[0], grid_size[1]
        self._goal_state = goal_state
        self._num_actions = num_actions
        self._k = k
        self._alpha = alpha
        self._gamma = gamma
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        
        self.C = self._build_costs()
        self.V = self._build_V0()
        self.Qi1 = self._build_Q0()
        self.Qi = self._build_Q0()
        
    def __repr__(self):
        self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                               str_title=f'Piecewise Linear VI - RF: {self._river_flow} / k: {self._k}')
        return '> Visualização da Política \n' + \
            f'k: {self._k} \n' + \
            f'alpha: {self._alpha} \n' + \
            f'gamma: {self._gamma} '
            
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
            
    def _build_V0(self):
        V0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                V0[(r, c)] = 0
        return V0
    
    def _build_Q0(self):
        Q0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                Q0[(r, c)] = {}
                for a in range(self._num_actions):
                    Q0[(r, c)][a] = 0
        return Q0
    
    def _build_costs(self):
        C = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                C[(r, c)] = 1
        C[self._goal_state] = 0
        
        return C
        
    def _reward_function(self, S, action):
        reward = self._costs[action]
        
        # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
        if S == (self._goal_state[0], self._goal_state[1]):
            reward = 0
        
        return reward
    
    def _get_transition(self, S, a):
        transition_matrix = self._transition_probabilities[a][S]
        t = np.array([v[1] for v in transition_matrix.items()])
        return t
    
    def _get_costs(self):
        C = np.array([c[1] for c in self.C.items()])
        return C
    
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
        
    def _get_values_from_dict(self, d):
        V = np.array([v[1] for v in d.items()])
        return V
        
    def run_converge(self):
        i, time = self.calculate_value()
        return i, time
    
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
                        q += self._transition_probabilities[a][S][S_next] * self.function_O(Qi1[S_next], Qi1[S][a], C)

                    self.Qi[S][a] = Qi1[S][a] + self._alpha * q
                    
                    # print(f'S: [{S}] / Qi[S]: [{Qi1[S]}] / C: [{C}] / Qp: [{self.Qi[S][a]}] / q: [{q}]')
                    
                self.V[S] = min(self.Qi[S].values())
            
            # print(f'--- V: {self.V} \n --- Qa: {Qi1} \n --- Qp {self.Qi}')
            self._i += 1
            # if self._i == 2:
            #     break
            
            if self.relative_residual(self._get_values_from_dict(self.V), self._get_values_from_dict(V_prev)):
                break
                
        return self._i, 0
        
    def function_O(self, Q1, Q2, C):
        Q1min = []
        for a_line in range(self._num_actions):
            Q1min.append(Q1[a_line])
        
        X = self._piecewise_linear_transformation((C + self._gamma * min(Q1min) - Q2))

        return X
        
    def relative_residual(self, V1, V2):
        residual = []
        for i in range(len(V1)):
            try:
                residual.append(abs((V1[i] - V2[i])/V2[i]))
            except:
                residual.append(np.inf)
        # print('Residual: ', max(residual), end='\r')
        return max(residual) <= self._epsilon
    
    