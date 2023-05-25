
import numpy as np, random, copy
import time

from rl_utils.VizTools import VizTools

class RS_ValueIteration:
    def __init__(self, grid_size, goal_state, transition_probabilities, costs, 
                 vl_lambda, num_actions=4, epsilon=0.001, river_flow=None) -> None:
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
        
        self.V = self._build_V0()
        self.PI = self._build_PI0(True, True)
        self.Qi = self._build_Q0()
        
        self._first_run = True
        self._i = 0
    
    def __repr__(self):
        self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                               str_title=f'Exponential Utility Function - RSMDP')
        
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
    
    def _build_V0(self):
        V0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                # Preenche o V0 com o negativo do sinal do Lambda
                V0[(r, c)] = -np.sign(self._lambda)
        return V0
    
    def _build_Q0(self):
        Q0 = {}
        for r in range(0, self._rows):
            for c in range(0, self._cols):
                Q0[(r, c)] = {}
                for a in range(self._num_actions):
                    Q0[(r, c)][a] = 0
        return Q0
    
    def _get_random_action(self):
        return int(random.choice([i for i in range(0, self._num_actions)]))
        
    def _cost_function(self, S, action):
        reward = self._costs[action]
        
        # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
        if S == (self._goal_state[0], self._goal_state[1]):
            reward = 0
        
        return reward
    
    def _get_transition(self, S, a):
        transition_matrix = self._transition_probabilities[a][S]
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
    
    def _get_values_from_dict(self, d):
        V = np.array([v[1] for v in d.items()])
        return V
    
    def verify_residual(self, V, V_ANT):
        res = np.max(np.abs(np.subtract(V, V_ANT)))
        # print(f'> Residual: {res} \n \n \n')
        return res < 2 * self._epsilon
    
    def relative_residual(self, V1, V2):
        residual = []
        for i in range(len(V1)):
            try:
                residual.append(abs((V1[i] - V2[i])/V2[i]))
            except:
                residual.append(np.inf)
        # print('~~~~~~~~~~~~>> Residual: ', max(residual), end='\r')
        return max(residual) <= self._epsilon
    
    def run_converge(self):
        start_time = time.time()
        self.calculate_value()
        
        return self._i, (time.time() - start_time)
        
    def calculate_value(self):
        while True:
            V_prev = copy.deepcopy(self.V)
            
            for S in self.V.keys():
                for a in range(self._num_actions):
                    q = 0
                    for S_next in self.V.keys():
                        q += self._transition_probabilities[a][S][S_next] * self.V[S_next]

                    C = self._cost_function(S, a)
                    self.Qi[S][a] = np.exp(self._lambda * C) * q
                    
                    # print(f'V[s"]: [{self.V[S_next]}] S: [{S}] / Qi[S]: [{Qi1[S]}] / C: [{C}] / Qp: [{self.Qi[S][a]}] / q: [{q}]')
                
                if S == self._goal_state:
                    self.V[S] = -np.sign(self._lambda)
                else:
                    self.V[S] = min(self.Qi[S].values())
                # print(f'V: [{self.V}]')
            self._i += 1
            
            if self.verify_residual(self._get_values_from_dict(self.V), self._get_values_from_dict(V_prev)):
                break
            
        # Compute the optimal policy
        for S in self.V.keys():
            for a in range(self._num_actions):
                q = 0
                for S_next in self.V.keys():
                    q += self._transition_probabilities[a][S][S_next] * self.V[S_next]

                C = self._cost_function(S, a)
                self.Qi[S][a] = np.exp(self._lambda * C) * q
                    
            self.PI[S] = min(self.Qi[S], key=self.Qi[S].get)
            
    def calculate_value_for_policy(self, Pi, vl_lambda):
        V = copy.deepcopy(self._build_V0())
        i = 0
        
        while True:
            accumulate_cost = 0
            V_prev = copy.deepcopy(V)
            
            for S in Pi.keys():
                a = Pi[S]
                
                q = sum(self._get_transition(S, a) * self._get_values_from_dict(V))
                # for S_next in Pi.keys():
                #     q += self._transition_probabilities[a][S][S_next] * V[S_next]

                C = self._cost_function(S, a)
                
                if S == self._goal_state:
                    V[S] = -np.sign(vl_lambda)
                    accumulate_cost += -np.sign(vl_lambda)
                else:
                    V[S] = np.exp(vl_lambda * C) * q
                    accumulate_cost += np.exp(vl_lambda * C) * q
            
            print(f'Exp: {np.exp(vl_lambda * C)} / q: {q} / Acc_Cost: [{accumulate_cost}] ')
            # print(f'V: {V} \n V_Prev: {V_prev} \n\n\n')
            i += 1
        
            if self.verify_residual(self._get_values_from_dict(V), self._get_values_from_dict(V_prev)):
                break
            
        return accumulate_cost