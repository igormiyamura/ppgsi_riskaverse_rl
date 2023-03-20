
import random
from rl_utils import UtilFunctions as uf

class AverageCosts:
    def __init__(self, model, costs, give_up=1000):
        self._rows, self._cols = model._rows, model._cols
        self._give_up = give_up
        
        self._initial_state = (0, 0)
        self._goal_state = model._goal_state
        
        self._policy = model.PI
        self._transitions = model._transition_probabilities
        self._costs = costs
        
    def _reward_function(self, S, action):
        reward = self._costs[action]
        
        # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
        if action == 2 and S == (self._goal_state[0], self._goal_state[1] + 1):
            reward = 0
        
        return reward
        
    def _build_cumulative_prob(self, t):
        res, cumulative_prob = {}, 0
        for k, v in t.items():
            cumulative_prob += v
            res[k] = cumulative_prob
        
        return res
    
    def _find_next_state(self, t_acum, x):
        for k, v in t_acum.items():
            if x <= v:
                return k
        
    def run_simulations(self, num_simulations):
        res_costs, res_it = {}, {}
        
        for i in range(num_simulations):
            state = self._initial_state
            cost_acum, iterations = 0, 0
            while (state != self._goal_state and iterations < self._give_up):
                action = self._policy[state]
                
                t = self._transitions[action][state]
                t = dict(filter(uf.filter_more_than_zero, t.items()))
                
                x = random.random()
                
                t_acum = self._build_cumulative_prob(t)
                next_state = self._find_next_state(t_acum, x)
                
                cost_acum += self._reward_function(state, action)
                iterations += 1
                
                state = next_state
            
            res_costs[i] = cost_acum
            res_it[i] = iterations
            
        return res_costs, res_it

