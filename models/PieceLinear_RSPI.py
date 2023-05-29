
import numpy as np, random
import time

from rl_utils.VizTools import VizTools

np.seterr(divide='ignore', invalid='ignore')

class PieceLinear_RSPI:
    def __init__(self, env, transition_probabilities, 
                 costs, k, alpha, gamma, num_actions=4, epsilon=0.001, river_flow=None) -> None:
        self.viz_tools = VizTools()
        self._env = env
        
        self._river_flow = river_flow
        self._grid_size = self._env._grid_size
        self._rows, self._cols = self._env._grid_size[0], self._env._grid_size[1]
        self._goal_state = self._env._goal_state
        self._num_actions = num_actions
        self._k = k
        self._alpha = alpha
        self._gamma = gamma
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        
        self.V = self._build_V0()
        self.V_ANT = self._build_V0()
        self.PI = self._build_PI0(True, False)
        self.C = self._build_costs()
        
        self._first_run = True
        self._i = 0
        
    def __repr__(self):
        self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                               str_title=f'Piecewise Linear PI - RF: {self._river_flow} / k: {self._k}')
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
    
    def _build_costs(self):
        C = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                C[(r, c)] = 1
        C[self._goal_state] = 0
        
        return C
    
    def _build_V0(self):
        V0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                V0[(r, c)] = 0
        return V0
    
    def _get_random_action(self):
        return int(random.choice([i for i in range(0, self._num_actions)]))
        
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
    
    def _get_costs(self):
        C = np.array([c[1] for c in self.C.items()])
        return C
    
    def _get_V(self):
        V = np.array([v[1] for v in self.V.items()])
        return V
    
    def _get_V_ant(self):
        V = np.array([v[1] for v in self.V_ANT.items()])
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
        
    def run_converge(self):
        start_time = time.time()

        while(self._first_run or (self.PI != self.PI_ANT)):
            print(f'Iteração: {self._i}', end='\r')
            self.step()
            
            self._first_run = False
            self._i += 1
            
        return self._i, (time.time() - start_time)
        
    def step(self):
        self.policy_evaluation()
        self.policy_improvement()
    
    def policy_evaluation(self):
        i = 0
        # self.V_ANT = self._build_V0()
        
        while(i == 0 or self.relative_residual(self._get_V(), self._get_V_ant())):
            # print('V:', self._get_V())
            # print('V_ANT:', self._get_V_ant())
            self.V_ANT = self.V.copy()
            
            for S in self.V.keys():
                a = self.PI[S]
                
                V_ATUAL = self._get_V()
                V_ANTERIOR = self._get_V_ant()
                T = self._get_transition(S, a)
                C = self._get_costs()

                bellman = self._alpha * self.function_O(V_ATUAL, V_ANTERIOR, T, C)
                # print(S, bellman)
                self.V[S] += bellman
            
            i += 1
            
        return self.V
    
    def policy_improvement(self):
        self.PI_ANT = self.PI.copy()
        
        pi_improved = {}
        for S in self.V.keys():
            bellman = {}
            # improve the current policy by doing  the following update for every s ∈ S
            for a in range(0, self._num_actions):
                
                V_ATUAL = self._get_V()
                V_ANT = self._get_V_ant()
                
                # print('V:', self._get_V())
                # print('V_ANT:', self._get_V_ant())
            
                T = self._get_transition(S, a)
                C = self._get_costs()
                
                b = self.function_O(V_ATUAL, V_ANT, T, C)
                    
                bellman[a] = b
                
            pi_improved[S] = min(bellman, key=bellman.get)
        # print('PI Atual:', self.PI)
        # print('PI Improved:', pi_improved)
        self.PI = pi_improved
        return self.PI
    
    def function_O(self, V_ATUAL, V_ANT, T, C):
        X = self._piecewise_linear_transformation((C + self._gamma * V_ATUAL - V_ANT))
        return (T * X).sum()
    
    def get_residual_rho(self):
        return (1 - self._alpha * (1 - abs(self._k)) * (1 - self._gamma))
    
    def relative_residual(self, V_ATUAL, V_ANTERIOR):
        residual = []
        for i in range(0, len(V_ATUAL)):
            try:
                residual.append(abs((V_ATUAL[i] - V_ANTERIOR[i])/V_ANTERIOR[i]))
            except:
                residual.append(np.inf)
        # print('Residual: ', max(residual))
        return max(residual) > self._epsilon
    
    def verify_residual(self, O_V1, O_V2, V1, V2):
        return abs(O_V1 - O_V2) <= self.get_residual_rho * abs(V1 - V2)