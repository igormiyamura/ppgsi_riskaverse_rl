
import numpy as np

class ErrorMetrics:
    def __init__(self, epsilon) -> None:
        self._epsilon = epsilon
    
    def absolute_residual(self, V, V_ANT, QUIET=True):
        res = np.max(np.abs( np.subtract(list(V.values()), list(V_ANT.values())) ))
        if not QUIET: print(f'>> Residual: {res}', end='\r')
        return res < 2 * self._epsilon
    
    def relative_residual(self, V1, V2, QUIET=True):
        residual = []
        
        for i in range(len(V1)):
            try:
                residual.append(abs((V1[i] - V2[i])/V2[i]))
            except:
                residual.append(np.inf)
                
        if not QUIET: print(f'>> Residual: {max(residual)}', end='\r')
        return max(residual) <= self._epsilon