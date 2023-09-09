
import random

class SimpleMDP:
    
    def __init__(self, num_states, num_actions, _fixed_probability = 0.8, _float_probability = 0.04) -> None:
        self._env_name = 'SimpleMDP'
        
        self._fixed_probability = _fixed_probability
        self._float_probability = _float_probability
        
        self._num_states = num_states
        self._num_actions = num_actions
        self._goal_state = 'sG'
        
        self._actions = [a for a in range(0, self._num_actions)]
        self._states = [s for s in range(0, self._num_states)]
        self._states.append(self._goal_state)
        
    def _approved_probability(self, x):
        return self._fixed_probability + self._float_probability * x
        
    def _verify_sum_probabilities(self, transition_probabilities):
        is_ok, dict_verification = True, {}
        for action in transition_probabilities.keys():
            for state in transition_probabilities[action].keys():
                dict_verification[(action, state)] = True if sum([v[1] for v in transition_probabilities[action][state].items()]) == 1 else False

                if dict_verification[(action, state)] == False:
                    print(action, state, [v[1] for v in transition_probabilities[action][state].items()])
                    is_ok = False
                        
        return is_ok, dict_verification
    
    def _build_V0(self, initial_value=0):
        V = {}
        
        for s in self._states:
            V[s] = initial_value
            
        return V
    
    def _build_PI0(self, initial_value=-1):
        PI = {}
        
        for s in self._states:
            PI[s] = initial_value
            
        return PI
    
    def _build_Q0(self, initial_value=0):
        Q0 = {}
        for s in self._states:
            Q0[s] = {}
            for a in self._actions:
                Q0[s][a] = 0
        return Q0
    
    def build_default_states_action_transition_dictionary(self, states, actions):
        res = {}
        
        for s in states:
            res[s] = {}
            for a in range(0, actions):
                res[s][a] = {}
                for s_next in states:
                    res[s][a][s_next] = 0
        
        return res
        
    def build_transition_probabilities(self):
        T = self.build_default_states_action_transition_dictionary(self._states, self._num_actions)
        
        for s in T.keys():
            for a in T[s].keys():
                if s == 'sG':
                    T[s][a]['sG'] = 1 # Caso esteja na meta, continua com probabilidade 1
                else:
                    T[s][a]['sG'] = self._approved_probability(a) # Vai para a meta
                    T[s][a][s] = 1 - self._approved_probability(a) # Permanece no mesmo lugar
        
        self._verify_sum_probabilities(T)
        
        return T