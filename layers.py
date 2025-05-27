import numpy as np
from typing import Optional, Dict, Tuple
from activation_function import ActivationFunction, BackPropActivationFunction


class EmbeddingLayer:
    
    def __init__(self, vocab_size:int, embedding_dim: int, weights: Optional[np.ndarray] = None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.cache = {}

        if weights is not None:
            self.weights = weights

        else:
            # initialize with small random values
            self.weights = np.random.uniform(-0.05, 0.05,(vocab_size, embedding_dim))
    
    def forward(self, x: np.ndarray) -> np.ndarray:

        self.cache['x'] = x

        if x.ndim == 1: 
            # Single Sequence
            return self.weights[x]
        elif x.ndim == 2:
            # Batch of Sequence
            batch_size, seq_lenght = x.shape
            embedded = np.zeros((batch_size, seq_lenght, self.embedding_dim))
            for i in range(batch_size):
                embedded[i] = self.weights[x[i]]
            return embedded
        else:
            raise ValueError(f"Invalid input dimenssion : {x.ndim}")
        
    def backward(self, dout: np.ndarray) -> Dict:

        x = self.cache['x']
        dW = np.zeros_like(self.weights)
        
        if x.ndim == 1:
            # Single sequence
            for i, idx in enumerate(x):
                dW[idx] += dout[i]
        elif x.ndim == 2:
            # Batch of sequences
            batch_size, seq_length = x.shape
            for b in range(batch_size):
                for s in range(seq_length):
                    idx = x[b, s]
                    dW[idx] += dout[b, s]
        
        return {'dW': dW}
    
    def get_parameters(self) -> Dict:
        return {'weights': self.weights}

    def set_parameters(self, params: Dict):
        self.weights = params['weights']

class DropoutLayer:
    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self.training = True
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0.0:
            self.cache['mask'] = np.ones_like(x)
            return x
        
        keep_prob = 1.0 - self.rate
        mask = np.random.binomial(1,keep_prob, x.shape)/ keep_prob
        self.cache['mask'] = mask
        return x * mask
    
    def set_training(self, training : bool):
        self.training = training

    def backward(self, dout: np.ndarray) -> np.ndarray:

        mask = self.cache['mask']
        return dout * mask 

class DenseLayer:
    
    def __init__(self, input_size:int, output_size: int, activation: str = 'linear', 
                 weights: Optional[Dict] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.cache = {}

        if weights is not None:
            self.W = weights['kernel']
            self.b = weights['bias'] 
        else:
            # Xavier initialization
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
            self.b = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:

        self.cache['x'] = x
        
        linear_output = np.dot(x, self.W) + self.b
        self.cache['linear_output'] = linear_output

        if self.activation == 'sigmoid':
            output = ActivationFunction.sigmoid(linear_output)
        elif self.activation == 'relu':
            output = ActivationFunction.relu(linear_output)
        elif self.activation == 'tanh':
            output = ActivationFunction.tanh(linear_output)
        elif self.activation == 'softmax':
            output = ActivationFunction.softmax(linear_output)
        else:
            output = linear_output
        
        self.cache['output'] = output
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict]:

        x = self.cache['x']
        linear_output = self.cache['linear_output']

        # Gradient through activation
        if self.activation == 'sigmoid':
            dlinear = dout * BackPropActivationFunction.sigmoid_derivative(linear_output)
        elif self.activation == 'relu':
            dlinear = dout * BackPropActivationFunction.relu_derative(linear_output)
        elif self.activation == 'tanh':
            dlinear = dout * BackPropActivationFunction.tanh_derivative(linear_output)
        elif self.activation == 'softmax':
            dlinear = dout * BackPropActivationFunction.softmax_derivative(linear_output)
        else:
            dlinear = dout

        # Gradients w.r.t weights and biases
        dW = np.dot(x.T, dlinear)
        db = np.sum(dlinear, axis=0)
        
        # Gradient w.r.t input
        dx = np.dot(dlinear, self.W.T)
        
        gradients = {'dW': dW, 'db': db}
        return dx, gradients
    
    def get_parameters(self) -> Dict:
        return {'W': self.W, 'b': self.b}

    def set_parameters(self, params: Dict):
        self.W = params['W']
        self.b = params['b']
    


        