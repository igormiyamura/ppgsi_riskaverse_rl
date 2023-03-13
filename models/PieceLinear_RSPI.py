
import numpy as np, random

class PieceLinear_RSPI:
    def __init__(self, grid_size, goal_state, transition_probabilities, costs, k, alpha, gamma, num_actions=4) -> None:
        self._rows, self._cols = grid_size[0], grid_size[1]
        self._goal_state = goal_state
        self._num_actions = num_actions
        self._k = k
        self._alpha = alpha
        self._gamma = gamma
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        
        self.V = self._build_V0()
        self.V_ANT = self._build_V0()
        self.PI = self._build_PI0(True, True)
        self._first_run = True
        self._i = 0
        
    def _verify_alpha(self):
        if (self._alpha <= 0) or (self._alpha > (1 + abs(self._k)) ** (-1)): raise Exception(f'Valor de alpha [{self._alpha}] inválido.')
    
    def _verify_k(self):
        if (self._k < -1) or (self._k > 1): raise Exception(f'Valor de k [{self._k}] inválido.')
    
    def _piecewise_linear_transformation(self, x):
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
        
        while(self._first_run or (self.PI != self.PI_ANT)):
            print(f'Iteração: {self._i}', end='\r')
            self.step()
            
            self._first_run = False
            self._i += 1
            
        return self._i
        
    def step(self):
        self.policy_evaluation()
        self.policy_improvement()
    
    def policy_evaluation(self):
        V = self.V
        
        for S in self.V.keys():
            a = self.PI[S]
            
            V_ATUAL = self._get_V()
            V_ANT = self._get_V_ant()
            T = self._get_transition(S, a)
            X = self._piecewise_linear_transformation(self._k)
            r = self._reward_function(S, a)
            # print(f'V_ATUAL: {V_ATUAL} / V_ANT: {V_ANT} / T: {T} / X: {X}, / r: {r}')
            
            bellman = self._alpha * \
                    (T * X * (r +  self._gamma * V_ATUAL - V_ANT)).sum()
            # print(f'S: {S} / Bellman: {bellman}')
            
            V[S] += bellman
        
        self.V = V
        self.V_ANT = self.V.copy()
        return self.V
    
    def policy_improvement(self):
        self.PI_ANT = self.PI
        
        pi_improved = {}
        for S in self.V.keys():
            bellman = {}
            # improve the current policy by doing  the following update for every s ∈ S
            for a in range(0, self._num_actions):
                
                V_ATUAL = self._get_V()
                V_ANT = self._get_V_ant()
                T = self._get_transition(S, a)
                X = self._piecewise_linear_transformation(self._k)
                r = self._reward_function(S, a)
                
                b = \
                    (T * X * (r + self._gamma * V_ATUAL - V_ANT)).sum()
                    
                bellman[a] = b
                
            pi_improved[S] = min(bellman, key=bellman.get)
        
        self.PI = pi_improved
        return self.PI
    
    