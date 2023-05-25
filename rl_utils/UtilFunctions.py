
import numpy as np

def filter_more_than_zero(pair):
    key, value = pair
    if value > 0:
        return True  # keep pair in the filtered dictionary
    else:
        return False  # filter pair out of the dictionary
    
def _get_values_from_dict(d):
    V = np.array([v[1] for v in d.items()])
    return V