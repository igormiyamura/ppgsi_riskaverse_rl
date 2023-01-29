

class RiverProblem:
    
    def __init__(self) -> None:
        pass
    
    def build_block_type(self, grid_size):
        block_type = {}
        t_row, t_col = grid_size[0], grid_size[1]
        for r in range(0, t_row): block_type[r] = {}
        
        for col in range(0, t_col):
            block_type[(0, col)] = 'river_bank'
            block_type[(t_row-1, col)] = 'river_bank'
            
        for row in range(1, t_row-1):
            block_type[(row, 0)] = 'waterfall'
            block_type[(row, t_col-1)] = 'bridge'
            
        for row in range(1, t_row-1):
            for col in range(1, t_col-1):
                block_type[(row, col)] = 'river'
                
        for r in range(0, t_row): block_type[r] = dict(sorted(block_type[r].items()))
        
        return block_type

    def build_default_transition_dictionary(self, num_actions):
        transition_prob = {}
        for a in range(0, num_actions):
            transition_prob[a] = {}
                
        return transition_prob

    def build_transition_probabilities(self, grid_size, block_type, river_flow=0.5):
        num_actions = 4
        t_row, t_col = grid_size[0], grid_size[1]
        transition_prob = self.build_default_transition_dictionary(num_actions)
        
        for action in range(0, num_actions):
            for row in range(0, t_row):
                for col in range(0, t_col):
                    # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
                    transition_prob[action][(row, col)] = {0: 0, 1: 0, 2: 0, 3: 0}
                    
                    if block_type[(row, col)] == 'river_bank' or block_type[(row, col)] == 'bridge': # Deterministico
                        transition_prob[action][(row, col)][action] = 1
                    elif block_type[(row, col)] == 'waterfall': # Dead end
                        transition_prob[action][(row, col)][2] = 1
                    elif block_type[(row, col)] == 'river':
                        if action != 2:
                            transition_prob[action][(row, col)][2] = river_flow
                            transition_prob[action][(row, col)][action] = (1 - river_flow)
                        else:
                            transition_prob[action][(row, col)][2] = 1
                    else:
                        raise '[build_transition_probabilities](!) Tipo de Bloco não identificado.'
                        
        return transition_prob

    def _verify_sum_probabilities(self, transition_probabilities):
        is_ok, dict_verification = True, {}
        for action in transition_probabilities.keys():
            for state in transition_probabilities[action].keys():
                dict_verification[(action, state)] = True if sum([v[1] for v in transition_probabilities[action][state].items()]) == 1 else False

                if dict_verification[(action, state)] == False:
                    is_ok = False
                        
        return is_ok, dict_verification
