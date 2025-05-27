import numpy as np
from typing import Dict

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step (global)
    
    def update(self, params: Dict, gradients: Dict, param_name: str):

        if param_name not in self.m:
            self.m[param_name] = {}
            self.v[param_name] = {}
            
            for key in params:
                self.m[param_name][key] = np.zeros_like(params[key])
                self.v[param_name][key] = np.zeros_like(params[key])
        
        self.t += 1
        
        for key in params:
            grad_key = f'd{key}' if f'd{key}' in gradients else f'd{key.upper()}'
            if grad_key not in gradients:
                continue
                
            grad = gradients[grad_key]
            
            # Update biased first moment estimate
            self.m[param_name][key] = self.beta1 * self.m[param_name][key] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[param_name][key] = self.beta2 * self.v[param_name][key] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[param_name][key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[param_name][key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
    
    def reset(self):
        self.t = 0
        self.m.clear()
        self.v.clear()