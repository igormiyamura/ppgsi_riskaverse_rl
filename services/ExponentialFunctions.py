
import numpy as np, math

class ExponentialFunctions:
    def __init__(self) -> None:
        pass
    
    def _utility(self, x, vl_lambda):
        return np.exp(vl_lambda * x)
    
    def _reverse_utility(self, x, vl_lambda):
        return math.log(x * np.sign(vl_lambda)) / vl_lambda