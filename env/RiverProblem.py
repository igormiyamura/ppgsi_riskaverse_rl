
import random, numpy as np

class RiverProblem:
    
    def __init__(self, grid_size, goal_state, dead_end=True) -> None:
        self._env_name = 'RiverProblem'
        
        self._num_actions = 4
        self._grid_size = grid_size
        self._t_row, self._t_col = self._grid_size[0], self._grid_size[1]
        self._goal_state = goal_state
        self._dead_end = dead_end
    
    def _get_random_action(self, num_actions):
        return int(random.choice([i for i in range(0, num_actions)]))
    
    def _build_PI0(self, initial_value=0, random=False, proper=False):
        PI0 = {}
        if proper:
            # Preenche todos os blocos de terra
            for c in range(0, self._t_col): 
                PI0[(0, c)] = 3
                PI0[(self._t_row - 1, c)] = 2
            for r in range(0, self._t_row - 1):
                PI0[(r, self._t_col - 1)] = 1
            # Preenche blocos waterfall
            for r in range(1, self._t_row-1):
                PI0[(r, 0)] = self.env._get_random_action(self._num_actions) if random else initial_value
            # Preenche todos os blocos de rio
            for r in range(1, self._t_row-1):
                for c in range(1, self._t_col-1):
                    PI0[(r, c)] = self.env._get_random_action(self._num_actions) if random else initial_value
        else:
            for r in range(0, self._t_row):
                for c in range(0, self._t_col):
                    PI0[(r, c)] = self.env._get_random_action(self._num_actions) if random else initial_value
        
        return PI0
    
    def _build_V0(self, initial_value=0):
        V0 = {}
        for r in range(0, self._t_row):
            for c in range(0, self._t_col):
                # Preenche o V0 com o negativo do sinal do Lambda
                V0[(r, c)] = initial_value
        return V0
    
    def _build_Q0(self, initial_value=0):
        Q0 = {}
        for r in range(0, self._t_row):
            for c in range(0, self._t_col):
                Q0[(r, c)] = {}
                for a in range(self._num_actions):
                    Q0[(r, c)][a] = initial_value
        return Q0
    
    def _next_state(self, state, action):
        x, y = state
        if action == 0:   # up
            x = max(x - 1, 0)
        elif action == 1: # down
            x = min(x + 1, self._t_row - 1)
        elif action == 2: # left
            y = max(y - 1, 0)
        elif action == 3: # right
            y = min(y + 1, self._t_col - 1)
        return (x, y)
    
    def build_block_type(self):
        block_type = {}
        
        for r in range(0, self._t_row): block_type[r] = {}
        
        for col in range(0, self._t_col):
            block_type[(0, col)] = 'river_bank'
            block_type[(self._t_row-1, col)] = 'river_bank'
            
        for row in range(1, self._t_row-1):
            block_type[(row, 0)] = 'waterfall'
            block_type[(row, self._t_col-1)] = 'bridge'
            
        for row in range(1, self._t_row-1):
            for col in range(1, self._t_col-1):
                block_type[(row, col)] = 'river'
                
        for r in range(0, self._t_row): block_type[r] = dict(sorted(block_type[r].items()))
        
        block_type[self._goal_state] = 'goal'

        return block_type

    def build_default_actions_transition_dictionary(self, num_actions):
        res = {}
        for a in range(0, num_actions):
            res[a] = {}
                
        return res
    
    def build_default_states_transition_dictionary(self, rows, cols):
        res = {}
        for row in range(0, rows):
            for col in range(0, cols):
                res[(row, col)] = 0
        return res
    
    def action_result(self, action, row, col):
        if action == 0:
            return max([row - 1, 0]), col
        elif action == 1:
            return min([row + 1, (self._t_row - 1)]), col
        elif action == 2:
            return row, max([col - 1, 0])
        elif action == 3:
            return row, min([col + 1, (self._t_col - 1)])
        else:
            raise Exception('Ação não definida.')

    def build_transition_probabilities(self, block_type, river_flow=0.5):
        num_actions = 4
        
        self.transition_prob = self.build_default_actions_transition_dictionary(num_actions)
        
        for action in range(0, num_actions):
            for row in range(0, self._t_row):
                for col in range(0, self._t_col):
                    # 0: DOWN, 1: UP, 2: LEFT, 3: RIGHT
                    self.transition_prob[action][(row, col)] = self.build_default_states_transition_dictionary(self._t_row, self._t_col) # {0: 0, 1: 0, 2: 0, 3: 0}
                    new_row, new_col = self.action_result(action, row, col)
                    
                    if block_type[(row, col)] == 'river_bank' or block_type[(row, col)] == 'bridge': # Deterministico
                        self.transition_prob[action][(row, col)][(new_row, new_col)] = 1
                    elif block_type[(row, col)] == 'waterfall': # dead_end = True or False
                        if self._dead_end:
                            self.transition_prob[action][(row, col)][(row, col)] = 1
                        else:
                            self.transition_prob[action][(row, col)][(0, 0)] = 1
                    elif block_type[(row, col)] == 'river':
                        if action != 2: # Probabilistico
                            self.transition_prob[action][(row, col)][(new_row, new_col)] = (1 - river_flow)
                            
                            left_row, left_col = self.action_result(2, row, col)
                            self.transition_prob[action][(row, col)][(left_row, left_col)] = river_flow
                        else: # Deterministico
                            self.transition_prob[action][(row, col)][(new_row, new_col)] = 1
                    elif block_type[(row, col)] == 'goal':
                        self.transition_prob[action][(row, col)][(row, col)] = 1
                        self.transition_prob[action][(row, col)][(new_row, new_col)] = 0
                    else:
                        raise '[build_self.transition_probabilities](!) Tipo de Bloco não identificado.'
                        
        return self.transition_prob

    def _verify_sum_probabilities(self, transition_probabilities, block_type):
        is_ok, dict_verification = True, {}
        for action in transition_probabilities.keys():
            for state in transition_probabilities[action].keys():
                dict_verification[(action, state)] = True if sum([v[1] for v in transition_probabilities[action][state].items()]) == 1 else False

                if dict_verification[(action, state)] == False and block_type[state] != 'goal':
                    print(action, state, [v[1] for v in transition_probabilities[action][state].items()])
                    is_ok = False
                        
        return is_ok, dict_verification
