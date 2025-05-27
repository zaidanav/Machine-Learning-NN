import numpy as np 

class ActivationFunction:

    @staticmethod
    def sigmoid(x: np.ndarray)-> np.ndarray:
        #Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x/np.sum(exp_x,axis=axis, keepdims=True)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        #Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)
    
class BackPropActivationFunction:
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        t = ActivationFunction.tanh(x)
        return 1 - t * t
    
    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunction.softmax(x)
        return s * (1 - s)
    
    @staticmethod
    def relu_derative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)