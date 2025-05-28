import numpy as np
from typing import Optional, Dict, Tuple, List
from activation_function import ActivationFunction, BackPropActivationFunction
from layers import EmbeddingLayer, DenseLayer, DropoutLayer
from optimizer import AdamOptimizer

class SimpleRNNCell:
    """
    Simple RNN Cell implementation from scratch
    
    RNN formula:
    h_t = tanh(W_hh * h_{t-1} + W_ih * x_t + b_h)
    """
    
    def __init__(self, input_size: int, hidden_size: int, weights: Optional[Dict] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if weights is not None:
            self.load_weights(weights)
        else:
            self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        # Input to hidden weights
        limit_ih = np.sqrt(6.0 / (self.input_size + self.hidden_size))
        self.W_ih = np.random.uniform(-limit_ih, limit_ih, (self.input_size, self.hidden_size))
        
        # Hidden to hidden weights  
        limit_hh = np.sqrt(6.0 / (self.hidden_size + self.hidden_size))
        self.W_hh = np.random.uniform(-limit_hh, limit_hh, (self.hidden_size, self.hidden_size))
        
        # Bias
        self.b_h = np.zeros(self.hidden_size)
    
    def load_weights(self, weights: Dict):
        """Load weights from dictionary"""
        self.W_ih = weights['W_ih']
        self.W_hh = weights['W_hh'] 
        self.b_h = weights['b_h']
    
    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass for single timestep
        
        Args:
            x_t: Input at time t, shape (batch_size, input_size)
            h_prev: Hidden state from previous timestep, shape (batch_size, hidden_size)
            
        Returns:
            h_t: Hidden state at time t, shape (batch_size, hidden_size)
            cache: Dictionary containing values needed for backward pass
        """
        # Linear transformation
        linear_output = np.dot(x_t, self.W_ih) + np.dot(h_prev, self.W_hh) + self.b_h
        
        # Apply tanh activation
        h_t = ActivationFunction.tanh(linear_output)
        
        # Cache values for backward pass
        cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'linear_output': linear_output,
            'h_t': h_t
        }
        
        return h_t, cache
    
    def backward_step(self, dh_t: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Backward pass for single timestep
        
        Args:
            dh_t: Gradient w.r.t hidden state at time t
            cache: Cached values from forward pass
            
        Returns:
            dx_t: Gradient w.r.t input at time t
            dh_prev: Gradient w.r.t previous hidden state
            gradients: Dictionary of parameter gradients
        """
        x_t = cache['x_t']
        h_prev = cache['h_prev']
        linear_output = cache['linear_output']
        
        # Gradient through tanh activation
        dlinear = dh_t * BackPropActivationFunction.tanh_derivative(linear_output)
        
        # Gradients w.r.t weights and bias
        dW_ih = np.dot(x_t.T, dlinear)
        dW_hh = np.dot(h_prev.T, dlinear)
        db_h = np.sum(dlinear, axis=0)
        
        # Gradients w.r.t inputs
        dx_t = np.dot(dlinear, self.W_ih.T)
        dh_prev = np.dot(dlinear, self.W_hh.T)
        
        gradients = {
            'dW_ih': dW_ih,
            'dW_hh': dW_hh,
            'db_h': db_h
        }
        
        return dx_t, dh_prev, gradients
    
    def get_parameters(self) -> Dict:
        """Get all parameters"""
        return {
            'W_ih': self.W_ih,
            'W_hh': self.W_hh,
            'b_h': self.b_h
        }
    
    def set_parameters(self, params: Dict):
        """Set parameters"""
        for key, value in params.items():
            setattr(self, key, value)


class SimpleRNNLayer:
    """
    Simple RNN Layer that processes sequences
    """
    
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = True,
                 go_backwards: bool = False, weights: Optional[Dict] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        
        self.rnn_cell = SimpleRNNCell(input_size, hidden_size, weights)
        self.cache = {}
    
    def forward(self, x: np.ndarray, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through RNN layer
        
        Args:
            x: Input sequences, shape (batch_size, seq_length, input_size)
            initial_state: Initial hidden state, shape (batch_size, hidden_size)
            
        Returns:
            output: RNN outputs
                - If return_sequences=True: (batch_size, seq_length, hidden_size)
                - If return_sequences=False: (batch_size, hidden_size)
        """
        batch_size, seq_length, _ = x.shape
        
        # Initialize hidden state
        if initial_state is not None:
            h_t = initial_state
        else:
            h_t = np.zeros((batch_size, self.hidden_size))
        
        # Determine sequence order
        if self.go_backwards:
            sequence_order = range(seq_length - 1, -1, -1)
        else:
            sequence_order = range(seq_length)
        
        # Process sequence
        outputs = []
        caches = []
        
        for t in sequence_order:
            x_t = x[:, t, :]
            h_t, cache = self.rnn_cell.forward_step(x_t, h_t)
            outputs.append(h_t.copy())
            caches.append(cache)
        
        # Store cache for backward pass
        self.cache = {
            'x': x,
            'caches': caches,
            'sequence_order': sequence_order,
            'outputs': outputs
        }
        
        if self.return_sequences:
            # Return all outputs in original temporal order
            if self.go_backwards:
                outputs = outputs[::-1]
            return np.stack(outputs, axis=1)
        else:
            # Return only last output
            return outputs[-1]
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Backward pass through RNN layer
        
        Args:
            dout: Gradient from next layer
                - If return_sequences=True: (batch_size, seq_length, hidden_size)
                - If return_sequences=False: (batch_size, hidden_size)
                
        Returns:
            dx: Gradient w.r.t input, shape (batch_size, seq_length, input_size)
            param_grads: Dictionary of parameter gradients
        """
        cache = self.cache
        x = cache['x']
        caches = cache['caches']
        sequence_order = cache['sequence_order']
        
        batch_size, seq_length, _ = x.shape
        
        # Initialize gradients
        dx = np.zeros_like(x)
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        # Initialize parameter gradients
        param_grads = {
            'dW_ih': np.zeros_like(self.rnn_cell.W_ih),
            'dW_hh': np.zeros_like(self.rnn_cell.W_hh),
            'db_h': np.zeros_like(self.rnn_cell.b_h)
        }
        
        # Prepare output gradients
        if self.return_sequences:
            if self.go_backwards:
                # Reverse dout to match computation order
                output_grads = dout[:, ::-1, :]
            else:
                output_grads = dout
        else:
            # Only last timestep has gradient
            output_grads = np.zeros((batch_size, len(sequence_order), self.hidden_size))
            output_grads[:, -1, :] = dout
        
        # Backward pass (reverse of forward computation order)
        for i in range(len(sequence_order) - 1, -1, -1):
            t = sequence_order[i]
            
            # Get gradient for current timestep
            dh_t = output_grads[:, i, :] + dh_next
            
            # Backward step
            dx_t, dh_next, grads = self.rnn_cell.backward_step(dh_t, caches[i])
            
            # Store input gradient
            dx[:, t, :] = dx_t
            
            # Accumulate parameter gradients
            for key in param_grads:
                param_grads[key] += grads[key]
        
        return dx, param_grads


class BidirectionalRNN:
    """
    Bidirectional RNN Layer
    """
    
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = True,
                 weights: Optional[Dict] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        
        # Initialize forward and backward RNN layers
        forward_weights = weights.get('forward', None) if weights else None
        backward_weights = weights.get('backward', None) if weights else None
        
        self.forward_rnn = SimpleRNNLayer(input_size, hidden_size, return_sequences=True,
                                        go_backwards=False, weights=forward_weights)
        self.backward_rnn = SimpleRNNLayer(input_size, hidden_size, return_sequences=True,
                                         go_backwards=True, weights=backward_weights)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through bidirectional RNN
        
        Args:
            x: Input sequences, shape (batch_size, seq_length, input_size)
            
        Returns:
            output: Concatenated forward and backward outputs
                - If return_sequences=True: (batch_size, seq_length, 2*hidden_size)
                - If return_sequences=False: (batch_size, 2*hidden_size)
        """
        # Forward and backward passes
        forward_output = self.forward_rnn.forward(x)
        backward_output = self.backward_rnn.forward(x)
        
        # Concatenate outputs
        output = np.concatenate([forward_output, backward_output], axis=-1)
        
        if not self.return_sequences:
            return output[:, -1, :]
        
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Backward pass through bidirectional RNN
        
        Args:
            dout: Gradient from next layer
            
        Returns:
            dx: Gradient w.r.t input
            combined_grads: Dictionary of parameter gradients
        """
        # Handle different input shapes
        if dout.ndim == 2:
            # If return_sequences=False, expand to sequence dimension
            batch_size, hidden_dim = dout.shape
            seq_length = self.forward_rnn.cache['x'].shape[1] if 'x' in self.forward_rnn.cache else 1
            dout_expanded = np.zeros((batch_size, seq_length, hidden_dim))
            dout_expanded[:, -1, :] = dout  # Only last timestep gets gradient
            dout = dout_expanded
        
        # Split gradients for forward and backward
        hidden_size = self.hidden_size
        dout_forward = dout[..., :hidden_size]
        dout_backward = dout[..., hidden_size:]
        
        # Backward passes
        dx_forward, grads_forward = self.forward_rnn.backward(dout_forward)
        dx_backward, grads_backward = self.backward_rnn.backward(dout_backward)
        
        # Combine input gradients
        dx = dx_forward + dx_backward
        
        # Combine parameter gradients
        combined_grads = {
            'forward': grads_forward,
            'backward': grads_backward
        }
        
        return dx, combined_grads


class SimpleRNNModel:
    """
    Complete Simple RNN Model for text classification with bonus features
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_units: int, num_classes: int,
                 num_rnn_layers: int = 1, bidirectional: bool = False, dropout_rate: float = 0.5,
                 return_sequences: bool = False, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.num_classes = num_classes
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        self.training = True
        
        self._build_model()
    
    def _build_model(self):
        """Build the RNN model"""
        # Embedding layer
        self.embedding = EmbeddingLayer(self.vocab_size, self.embedding_dim)
        
        # RNN layers
        self.rnn_layers = []
        for i in range(self.num_rnn_layers):
            input_size = self.embedding_dim if i == 0 else (
                self.rnn_units * 2 if self.bidirectional else self.rnn_units
            )
            
            return_seq = True if i < self.num_rnn_layers - 1 else self.return_sequences
            
            if self.bidirectional:
                rnn_layer = BidirectionalRNN(input_size, self.rnn_units, return_sequences=return_seq)
            else:
                rnn_layer = SimpleRNNLayer(input_size, self.rnn_units, return_sequences=return_seq)
            
            self.rnn_layers.append(rnn_layer)
        
        # Dropout layer
        self.dropout = DropoutLayer(self.dropout_rate)
        
        # Dense layer
        dense_input_size = self.rnn_units * 2 if self.bidirectional else self.rnn_units
        self.dense = DenseLayer(dense_input_size, self.num_classes, activation='softmax')
    
    def forward(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Forward pass with batch support (Bonus 2)
        
        Args:
            x: Input sequences, shape (batch_size, seq_length) or (seq_length,)
            batch_size: If provided, process in batches for large inputs
            
        Returns:
            output: Predictions, shape (batch_size, num_classes)
        """
        # Handle single sequence input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Handle batch inference (Bonus 2)
        if batch_size is not None and x.shape[0] > batch_size:
            return self._batch_forward(x, batch_size)
        
        # Embedding layer
        embedded = self.embedding.forward(x)
        
        # RNN layers
        rnn_output = embedded
        for rnn_layer in self.rnn_layers:
            rnn_output = rnn_layer.forward(rnn_output)
        
        # Dropout layer
        dropped = self.dropout.forward(rnn_output)
        
        # Dense layer
        output = self.dense.forward(dropped)
        
        return output
    
    def backward(self, dout: np.ndarray) -> Dict:
        """
        Backward pass (Bonus 1)
        
        Args:
            dout: Gradient from loss function
            
        Returns:
            gradients: Dictionary of all gradients
        """
        # Backward through dense layer
        ddropout, dense_grads = self.dense.backward(dout)
        
        # Backward through dropout
        drnn_final = self.dropout.backward(ddropout)
        
        # Backward through RNN layers
        drnn = drnn_final
        rnn_grads = []
        for i in reversed(range(len(self.rnn_layers))):
            drnn, rnn_grad = self.rnn_layers[i].backward(drnn)
            rnn_grads.insert(0, rnn_grad)
        
        # Backward through embedding
        embedding_grads = self.embedding.backward(drnn)
        
        return {
            'embedding_grads': embedding_grads,
            'rnn_grads': rnn_grads,
            'dense_grads': dense_grads
        }
    
    def _batch_forward(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Process large inputs in batches (Bonus 2)
        """
        num_samples = x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        results = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]
            
            batch_output = self.forward(batch_x)
            results.append(batch_output)
        
        return np.vstack(results)
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute sparse categorical crossentropy loss"""
        batch_size = predictions.shape[0]
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        correct_class_probs = predictions[np.arange(batch_size), targets]
        loss = -np.mean(np.log(correct_class_probs))
        
        return loss
    
    def compute_loss_gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of loss w.r.t predictions"""
        batch_size = predictions.shape[0]
        dout = predictions.copy()
        dout[np.arange(batch_size), targets] -= 1
        dout /= batch_size
        
        return dout
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Single training step with backward propagation (Bonus 1)"""
        # Forward pass
        predictions = self.forward(x)
        
        # Compute loss
        loss = self.compute_loss(predictions, y)
        
        # Backward pass (Bonus 1)
        loss_grad = self.compute_loss_gradient(predictions, y)
        gradients = self.backward(loss_grad)
        
        # Update weights
        self._update_weights(gradients)
        
        return loss
    
    def _update_weights(self, gradients: Dict):
        """Update model weights using optimizer"""
        # Update embedding weights
        embedding_params = self.embedding.get_parameters()
        self.optimizer.update(embedding_params, gradients['embedding_grads'], 'embedding')
        
        # Update RNN weights
        for i, rnn_grads in enumerate(gradients['rnn_grads']):
            if self.bidirectional:
                # Update forward RNN
                forward_params = self.rnn_layers[i].forward_rnn.rnn_cell.get_parameters()
                forward_grads = rnn_grads['forward']
                self.optimizer.update(forward_params, forward_grads, f'rnn_{i}_forward')
                
                # Update backward RNN
                backward_params = self.rnn_layers[i].backward_rnn.rnn_cell.get_parameters()
                backward_grads = rnn_grads['backward']
                self.optimizer.update(backward_params, backward_grads, f'rnn_{i}_backward')
            else:
                # Update unidirectional RNN
                rnn_params = self.rnn_layers[i].rnn_cell.get_parameters()
                self.optimizer.update(rnn_params, rnn_grads, f'rnn_{i}')
        
        # Update dense weights
        dense_params = self.dense.get_parameters()
        self.optimizer.update(dense_params, gradients['dense_grads'], 'dense')
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32, verbose: bool = True) -> Dict:
        """
        Train the model with batch support
        """
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.set_training(True)
            train_losses = []
            train_correct = 0
            train_total = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training with batch support (Bonus 2)
            for i in range(0, len(X_shuffled), batch_size):
                batch_x = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Training step with backward propagation (Bonus 1)
                loss = self.train_step(batch_x, batch_y)
                train_losses.append(loss)
                
                # Calculate accuracy
                predictions = self.predict(batch_x)
                train_correct += np.sum(predictions == batch_y)
                train_total += len(batch_y)
            
            # Calculate training metrics
            train_loss = np.mean(train_losses)
            train_acc = train_correct / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                self.set_training(False)
                val_predictions = self.forward(X_val, batch_size=batch_size)
                val_loss = self.compute_loss(val_predictions, y_val)
                val_pred_classes = np.argmax(val_predictions, axis=-1)
                val_acc = np.mean(val_pred_classes == y_val)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}")
        
        return history
    
    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Make predictions with batch support (Bonus 2)"""
        self.set_training(False)
        predictions = self.forward(x, batch_size)
        return np.argmax(predictions, axis=-1)
    
    def predict_proba(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Get prediction probabilities with batch support (Bonus 2)"""
        self.set_training(False)
        return self.forward(x, batch_size)
    
    def set_training(self, training: bool):
        """Set training mode"""
        self.training = training
        self.dropout.set_training(training)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32) -> Dict:
        """Evaluate model with batch support (Bonus 2)"""
        from sklearn.metrics import f1_score, classification_report
        
        self.set_training(False)
        
        # Get predictions with batch processing
        predictions_proba = self.forward(X_test, batch_size=batch_size)
        predictions = np.argmax(predictions_proba, axis=-1)
        
        # Calculate metrics
        test_loss = self.compute_loss(predictions_proba, y_test)
        test_acc = np.mean(predictions == y_test)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'macro_f1_score': macro_f1,
            'predictions': predictions,
            'classification_report': classification_report(y_test, predictions)
        }
        
        return results
    
    def save_weights(self, filepath: str):
        """Save model weights"""
        weights = {
            'embedding': self.embedding.get_parameters(),
            'rnn_layers': [],
            'dense': self.dense.get_parameters()
        }
        
        for i, rnn_layer in enumerate(self.rnn_layers):
            if self.bidirectional:
                layer_weights = {
                    'forward': rnn_layer.forward_rnn.rnn_cell.get_parameters(),
                    'backward': rnn_layer.backward_rnn.rnn_cell.get_parameters()
                }
            else:
                layer_weights = rnn_layer.rnn_cell.get_parameters()
            
            weights['rnn_layers'].append(layer_weights)
        
        np.save(filepath, weights)
    
    def load_weights(self, filepath: str):
        """Load model weights"""
        weights = np.load(filepath, allow_pickle=True).item()
        
        # Load embedding weights
        self.embedding.weights = weights['embedding']['weights']
        
        # Load RNN weights
        for i, layer_weights in enumerate(weights['rnn_layers']):
            if self.bidirectional:
                # Load forward RNN weights
                forward_weights = layer_weights['forward']
                for param_name, param_value in forward_weights.items():
                    setattr(self.rnn_layers[i].forward_rnn.rnn_cell, param_name, param_value)
                
                # Load backward RNN weights
                backward_weights = layer_weights['backward']
                for param_name, param_value in backward_weights.items():
                    setattr(self.rnn_layers[i].backward_rnn.rnn_cell, param_name, param_value)
            else:
                # Load unidirectional RNN weights
                for param_name, param_value in layer_weights.items():
                    setattr(self.rnn_layers[i].rnn_cell, param_name, param_value)
        
        # Load dense weights
        self.dense.W = weights['dense']['W']
        self.dense.b = weights['dense']['b']