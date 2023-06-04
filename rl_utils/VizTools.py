
import sys, pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class VizTools:
    def __init__(self) -> None:
        pass
    
    def visualize_policy_arrows(self, model_obj, V, goal_state, num_actions=4, type='V'):
        arrow_lenght = {
            0: [0, -0.3],
            1: [0, 0.3],
            2: [-0.3, 0],
            3: [0.3, 0]
        }
        
        if type == 'V':
            for S in V.keys():
                if S[0] != 0 and S[1] == 0: continue
                    
                best_action = -1
                action_values = {}
                for action in range(0, num_actions):
                    next_state = model_obj.env._next_state(S, action)
                    if next_state != S:
                        action_values[action] = V[next_state]
                        
                best_action = min(action_values, key=action_values.get)
                
                plt.arrow(S[1] + 0.5, S[0] + 0.5, arrow_lenght[best_action][0], arrow_lenght[best_action][1], fc="k", ec="k", head_width=0.06, head_length=0.1)
        elif type == 'PI':
            for S in V.keys():
                if S[0] != 0 and S[1] == 0: continue
                
                plt.arrow(S[1] + 0.5, S[0] + 0.5, arrow_lenght[V[S]][0], arrow_lenght[V[S]][1], fc="k", ec="k", head_width=0.06, head_length=0.1)

    def visualize_V(self, model_obj, V, grid_size, num_actions, goal_state, num_iterations, str_title, annot=True):
        df = pd.DataFrame(np.zeros(grid_size))
        for row in range(0, grid_size[0]):
            for col in range(0, grid_size[1]):
                df.loc[row, col] = np.round(V[(row, col)], 3)
                
        f, a = plt.subplots(figsize=(18, 6))
        f = plt.title(f'{str_title} - {num_iterations} it.', fontsize = 16)
        self.visualize_policy_arrows(model_obj, V, goal_state, num_actions)
        sns.heatmap(df, cmap="crest_r") # annot=annot,
        plt.show()
        return True
    
    def visualize_PI(self, model_obj, PI, grid_size, num_actions, goal_state, num_iterations, str_title, annot=True):
        df = pd.DataFrame(np.zeros(grid_size))
        
        f, a = plt.subplots(figsize=(18, 6))
        f = plt.title(f'{str_title} - {num_iterations} it.', fontsize = 16)
        
        ax.add_patch(Rectangle((3, 4), 1, 1, fill=False, edgecolor='blue', lw=3))
        ax.add_patch(Rectangle((3, 4), 1, 1, fill=False, edgecolor='blue', lw=3))
        
        self.visualize_policy_arrows(model_obj, PI, goal_state, num_actions, 'PI')
        sns.heatmap(df, cmap="crest_r") # annot=annot,
        plt.show()
        return True