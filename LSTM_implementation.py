import numpy as np
from activation_function import ActivationFunction, BackPropActivationFunction
from typing import Optional, Dict, Tuple, List
from layers import EmbeddingLayer, DenseLayer, DropoutLayer
from optimizer import AdamOptimizer

class LSTMCell:
    
    def __init__(self, input_size: int, hidden_size: int, weights: Optional[Dict] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # store forward pass values for backprop

        if weights is not None:
            self.load_weights(weights)
        else:
            self.initialize_weights()
    
    def initialize_weights(self):
        
        limit = np.sqrt(6.0 / (self.input_size + self.hidden_size))
    
        # Weight matrices for input-to-hidden connections
        self.W_i = np.random.uniform(-limit, limit, (self.input_size, self.hidden_size))
        self.W_f = np.random.uniform(-limit, limit, (self.input_size, self.hidden_size))
        self.W_c = np.random.uniform(-limit, limit, (self.input_size, self.hidden_size))
        self.W_o = np.random.uniform(-limit, limit, (self.input_size, self.hidden_size))

        # Weight matrices for hidden-to-hidden connections
        self.U_i = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))
        self.U_f = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))
        self.U_c = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))
        self.U_o = np.random.uniform(-limit, limit, (self.hidden_size, self.hidden_size))

        # Bias vectors
        self.b_i = np.zeros(self.hidden_size)
        self.b_f = np.zeros(self.hidden_size)
        self.b_c = np.zeros(self.hidden_size)
        self.b_o = np.zeros(self.hidden_size)

    def load_weights(self, weights: Dict):
        self.W_i = weights['W_i']
        self.W_f = weights['W_f']
        self.W_c = weights['W_c']
        self.W_o = weights['W_o']
        
        self.U_i = weights['U_i']
        self.U_f = weights['U_f']
        self.U_c = weights['U_c']
        self.U_o = weights['U_o']
        
        self.b_i = weights['b_i']
        self.b_f = weights['b_f']
        self.b_c = weights['b_c']
        self.b_o = weights['b_o']

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:

        # Input gate
        i_linear = np.dot(x_t, self.W_i) + np.dot(h_prev, self.U_i) + self.b_i
        i_t = ActivationFunction.sigmoid(i_linear)

        # Forget gate
        f_linear = np.dot(x_t, self.W_f) + np.dot(h_prev, self.U_f) + self.b_f
        f_t = ActivationFunction.sigmoid(f_linear)

        # Candidate values
        c_linear = np.dot(x_t, self.W_c) + np.dot(h_prev, self.U_c) + self.b_c
        c_tilde = ActivationFunction.tanh(c_linear)

        # Cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Output gate
        o_linear = np.dot(x_t, self.W_o) + np.dot(h_prev, self.U_o) + self.b_o
        o_t = ActivationFunction.sigmoid(o_linear)

        # Hidden state
        c_tanh = ActivationFunction.tanh(c_t)
        h_t = o_t * c_tanh

        timestep_cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'i_linear': i_linear,
            'f_linear': f_linear,
            'c_linear': c_linear,
            'o_linear': o_linear,
            'i_t': i_t,
            'f_t': f_t,
            'c_tilde': c_tilde,
            'o_t': o_t,
            'c_t': c_t,
            'c_tanh': c_tanh,
            'h_t': h_t
        }

        return h_t, c_t, timestep_cache 

    def backward_step(self, dh_t: np.ndarray, dc_t: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    
        x_t = cache['x_t']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        i_t = cache['i_t']
        f_t = cache['f_t']
        c_tilde = cache['c_tilde']
        o_t = cache['o_t']
        c_t = cache['c_t']
        c_tanh = cache['c_tanh']

        # Gradient for output gate
        do_t = dh_t * c_tanh
        do_linear = do_t * BackPropActivationFunction.sigmoid_derivative(cache['o_linear'])

        # Gradient for cell state from hidden state
        dc_t += dh_t * o_t * BackPropActivationFunction.tanh_derivative(c_t)

        # Gradient for forget gate
        df_t = dc_t * c_prev
        df_linear = df_t * BackPropActivationFunction.sigmoid_derivative(cache['f_linear'])
        
        # Gradient for input gate
        di_t = dc_t * c_tilde
        di_linear = di_t * BackPropActivationFunction.sigmoid_derivative(cache['i_linear'])
        
        # Gradient for candidate values
        dc_tilde = dc_t * i_t
        dc_linear = dc_tilde * BackPropActivationFunction.tanh_derivative(cache['c_linear'])
        
        # Gradient for previous cell state
        dc_prev = dc_t * f_t

        # Gradients for weights and biases (corrected matrix operations)
        dW_i = np.dot(x_t.T, di_linear)
        dW_f = np.dot(x_t.T, df_linear)
        dW_c = np.dot(x_t.T, dc_linear)
        dW_o = np.dot(x_t.T, do_linear)
        
        dU_i = np.dot(h_prev.T, di_linear)
        dU_f = np.dot(h_prev.T, df_linear)
        dU_c = np.dot(h_prev.T, dc_linear)
        dU_o = np.dot(h_prev.T, do_linear)
        
        # For biases (sum over batch dimension)
        db_i = np.sum(di_linear, axis=0)
        db_f = np.sum(df_linear, axis=0)
        db_c = np.sum(dc_linear, axis=0)
        db_o = np.sum(do_linear, axis=0)
        
        # Gradients for inputs
        dx_t = (np.dot(di_linear, self.W_i.T) + 
                np.dot(df_linear, self.W_f.T) + 
                np.dot(dc_linear, self.W_c.T) + 
                np.dot(do_linear, self.W_o.T))
        
        dh_prev = (np.dot(di_linear, self.U_i.T) + 
                np.dot(df_linear, self.U_f.T) + 
                np.dot(dc_linear, self.U_c.T) + 
                np.dot(do_linear, self.U_o.T))
        
        gradients = {
            'dW_i': dW_i, 'dW_f': dW_f, 'dW_c': dW_c, 'dW_o': dW_o,
            'dU_i': dU_i, 'dU_f': dU_f, 'dU_c': dU_c, 'dU_o': dU_o,
            'db_i': db_i, 'db_f': db_f, 'db_c': db_c, 'db_o': db_o
        }
        
        return dx_t, dh_prev, dc_prev, gradients
    
    def get_parameters(self) -> Dict:
        return {
            'W_i': self.W_i, 'W_f': self.W_f, 'W_c': self.W_c, 'W_o': self.W_o,
            'U_i': self.U_i, 'U_f': self.U_f, 'U_c': self.U_c, 'U_o': self.U_o,
            'b_i': self.b_i, 'b_f': self.b_f, 'b_c': self.b_c, 'b_o': self.b_o
        }

    def set_parameters(self, params: Dict):
        for key, value in params.items():
            setattr(self, key, value)

    
class LSTMLayer:
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = True, 
                 go_backwards: bool = False, weights: Optional[Dict] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self.lstm_cell = LSTMCell(input_size, hidden_size, weights)
        self.cache = {}
    
    def forward(self, x: np.ndarray, initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        batch_size, seq_length, _ = x.shape
        
        # Initial states
        if initial_state is not None:
            h_t, c_t = initial_state
        else:
            h_t = np.zeros((batch_size, self.hidden_size))
            c_t = np.zeros((batch_size, self.hidden_size))
        
        # Process sequence
        if self.go_backwards:
            sequence_order = range(seq_length - 1, -1, -1)  # seq_length-1, seq_length-2, ..., 0
        else:
            sequence_order = range(seq_length)  # 0, 1, 2, ..., seq_length-1
        
        outputs = []
        caches = []
        
        for t in sequence_order:
            x_t = x[:, t, :]
            h_t, c_t, cache = self.lstm_cell.forward_step(x_t, h_t, c_t)
            outputs.append(h_t.copy())
            caches.append(cache)
        
        # Store for backward pass
        self.cache = {
            'x': x,
            'caches': caches,
            'sequence_order': sequence_order,
            'outputs': outputs
        }
        
        if self.return_sequences:
            # Return all outputs in original temporal order
            if self.go_backwards:
                outputs = outputs[::-1]  # Reverse to match input order
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]  # Last computed output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict]:
        cache = self.cache
        x = cache['x']
        caches = cache['caches']
        sequence_order = cache['sequence_order']
        
        batch_size, seq_length, _ = x.shape
        
        # Initialize gradients
        dx = np.zeros_like(x)
        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # Initialize parameter gradients
        param_grads = {
            'dW_i': np.zeros_like(self.lstm_cell.W_i),
            'dW_f': np.zeros_like(self.lstm_cell.W_f),
            'dW_c': np.zeros_like(self.lstm_cell.W_c),
            'dW_o': np.zeros_like(self.lstm_cell.W_o),
            'dU_i': np.zeros_like(self.lstm_cell.U_i),
            'dU_f': np.zeros_like(self.lstm_cell.U_f),
            'dU_c': np.zeros_like(self.lstm_cell.U_c),
            'dU_o': np.zeros_like(self.lstm_cell.U_o),
            'db_i': np.zeros_like(self.lstm_cell.b_i),
            'db_f': np.zeros_like(self.lstm_cell.b_f),
            'db_c': np.zeros_like(self.lstm_cell.b_c),
            'db_o': np.zeros_like(self.lstm_cell.b_o)
        }
        
        # Prepare output gradients
        if self.return_sequences:
            if self.go_backwards:
                # Reverse dout to match computation order
                output_grads = dout[:, ::-1, :]
            else:
                output_grads = dout
        else:
            output_grads = np.zeros((batch_size, len(sequence_order), self.hidden_size))
            output_grads[:, -1, :] = dout  # Only last output has gradient
        
        # Backward pass (reverse of forward computation order)
        for i in range(len(sequence_order) - 1, -1, -1):
            t = sequence_order[i]
            
            # Get gradient for current timestep
            dh_t = output_grads[:, i, :] + dh_next
            
            # Backward step
            dx_t, dh_next, dc_next, grads = self.lstm_cell.backward_step(dh_t, dc_next, caches[i])
            
            # Store gradients
            dx[:, t, :] = dx_t
            for key in param_grads:
                param_grads[key] += grads[key]
        
        return dx, param_grads
        
class BidirectionalLSTM:

    def __init__(self, input_size: int, hidden_size: int,return_sequence: bool = True,
                weights: Optional[Dict] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequence = return_sequence

        forward_weights = weights.get('forward', None) if weights else None
        backward_weights = weights.get('backward', None) if weights else None

        # Initialize normal layer and reversed layer 
        self.forward_lstm = LSTMLayer(input_size, hidden_size, return_sequences=True, 
                                      go_backwards=False, weights=forward_weights)
        self.backward_lstm = LSTMLayer(input_size, hidden_size, return_sequences=True,
                                       go_backwards=False, weights=backward_weights)

    def forward(self, x:np.ndarray) -> np.ndarray:
        
        # find output
        forward_output = self.forward_lstm.forward(x)
        backward_output = self.backward_lstm.forward(x) 
        
        # Combine output
        output = np.concatenate([forward_output, backward_output], axis=-1)

        if not self.return_sequence:
            return output[:,-1, :]
        
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict]:

        # Split gradients for forward and backward
        hidden_size = self.hidden_size
        dout_forward = dout[..., :hidden_size]
        dout_backward = dout[..., hidden_size:]
        
        # Backward pass for both directions
        dx_forward, grads_forward = self.forward_lstm.backward(dout_forward)
        dx_backward, grads_backward = self.backward_lstm.backward(dout_backward)
        
        # Combine input gradients
        dx = dx_forward + dx_backward
        
        # Combine weight gradients
        combined_grads = {
            'forward': grads_forward,
            'backward': grads_backward
        }
        
        return dx, combined_grads
    
class LSTMModel:

    def __init__(self, vocab_size: int, embedding_dim: int, lstm_units:int, num_classes: int
                 , num_lstm_layers: int = 1, bidirectional: bool = False, dropout_rate: float = 0.5
                 , return_sequences:bool = False, learning_rate: float = 0.001,beta1: float = 0.9,
                   beta2: float = 0.999, epsilon: float = 1e-8):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        self.learning_rate = learning_rate


        self.optimizer = AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        self.training = True

        self._build_model()

    def _build_model(self):
        
        self.embedding = EmbeddingLayer(self.vocab_size, self.embedding_dim)

        self.lstm_layers = []
        for i in range(self.num_lstm_layers):
            input_size = self.embedding_dim if i == 0 else (
                self.lstm_units * 2 if self.bidirectional else self.lstm_units
            )

            return_seq = True if i < self.num_lstm_layers - 1 else self.return_sequences

            if self.bidirectional:
                lstm_layer = BidirectionalLSTM(input_size, self.lstm_units, return_sequence=return_seq)
            else:
                lstm_layer = LSTMLayer(input_size, self.lstm_units, return_sequences=return_seq)

            self.lstm_layers.append(lstm_layer)
        
        # Dropout Layer
        self.dropout = DropoutLayer(self.dropout_rate)

        # Dense Layer
        dense_input_size = self.lstm_units * 2 if self.bidirectional else self.lstm_units
        self.dense = DenseLayer(dense_input_size, self.num_classes, activation='softmax')

    def forward(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        
        # Handle single sequence input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Handle batch inference
        if batch_size is not None and x.shape[0] > batch_size:
            return self._batch_forward(x, batch_size)
        
        # Embedding layer
        embedded = self.embedding.forward(x)
        
        # LSTM layers
        lstm_output = embedded
        for lstm_layer in self.lstm_layers:
            lstm_output = lstm_layer.forward(lstm_output)
        
        # Dropout layer
        dropped = self.dropout.forward(lstm_output)
        
        # Dense layer
        output = self.dense.forward(dropped)
        
        return output
    
    def backward(self, dout: np.ndarray) -> Dict:
        # Backward through dense layer
        ddropout, dense_grads = self.dense.backward(dout)
        
        # Backward through dropout
        dlstm_final = self.dropout.backward(ddropout)
        
        # Backward through LSTM layers
        dlstm = dlstm_final
        lstm_grads = []
        for i in reversed(range(len(self.lstm_layers))):
            dlstm, lstm_grad = self.lstm_layers[i].backward(dlstm)
            lstm_grads.insert(0, lstm_grad)
        
        # Backward through embedding
        embedding_grads = self.embedding.backward(dlstm)
        
        return {
            'embedding_grads': embedding_grads,
            'lstm_grads': lstm_grads,
            'dense_grads': dense_grads
        }
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:

        batch_size = predictions.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Extract predicted probabilities for true classes
        correct_class_probs = predictions[np.arange(batch_size), targets]
        loss = -np.mean(np.log(correct_class_probs))
        
        return loss
    
    def compute_loss_gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        
        batch_size = predictions.shape[0]
        dout = predictions.copy()
        dout[np.arange(batch_size), targets] -= 1
        dout /= batch_size
        
        return dout

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        # Forward pass
        predictions = self.forward(x)
        
        # Compute loss
        loss = self.compute_loss(predictions, y)
        
        # Backward pass
        loss_grad = self.compute_loss_gradient(predictions, y)
        gradients = self.backward(loss_grad)
        
        # Update weights using Adam optimizer
        self._update_weights(gradients)
        
        return loss

    def _update_weights(self, gradients: Dict):

        # Update embedding weights
        embedding_params = self.embedding.get_parameters()
        self.optimizer.update(embedding_params, gradients['embedding_grads'], 'embedding')
        
        # Update LSTM weights
        for i, lstm_grads in enumerate(gradients['lstm_grads']):
            if self.bidirectional:
                # Update forward LSTM
                forward_params = self.lstm_layers[i].forward_lstm.lstm_cell.get_parameters()
                forward_grads = lstm_grads['forward']
                self.optimizer.update(forward_params, forward_grads, f'lstm_{i}_forward')
                
                # Update backward LSTM
                backward_params = self.lstm_layers[i].backward_lstm.lstm_cell.get_parameters()
                backward_grads = lstm_grads['backward']
                self.optimizer.update(backward_params, backward_grads, f'lstm_{i}_backward')
            else:
                # Update unidirectional LSTM
                lstm_params = self.lstm_layers[i].lstm_cell.get_parameters()
                self.optimizer.update(lstm_params, lstm_grads, f'lstm_{i}')
        
        # Update dense weights
        dense_params = self.dense.get_parameters()
        self.optimizer.update(dense_params, gradients['dense_grads'], 'dense')

    def _update_lstm_cell_weights(self, lstm_cell: LSTMCell, grads: Dict):
        
        lstm_cell.W_i -= self.learning_rate * grads['dW_i']
        lstm_cell.W_f -= self.learning_rate * grads['dW_f']
        lstm_cell.W_c -= self.learning_rate * grads['dW_c']
        lstm_cell.W_o -= self.learning_rate * grads['dW_o']
        
        lstm_cell.U_i -= self.learning_rate * grads['dU_i']
        lstm_cell.U_f -= self.learning_rate * grads['dU_f']
        lstm_cell.U_c -= self.learning_rate * grads['dU_c']
        lstm_cell.U_o -= self.learning_rate * grads['dU_o']
        
        lstm_cell.b_i -= self.learning_rate * grads['db_i']
        lstm_cell.b_f -= self.learning_rate * grads['db_f']
        lstm_cell.b_c -= self.learning_rate * grads['db_c']
        lstm_cell.b_o -= self.learning_rate * grads['db_o']

    def _batch_forward(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        
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
    
    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        self.set_training(False)
        predictions = self.forward(x, batch_size)
        return np.argmax(predictions, axis=-1)
    
    def predict_proba(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        self.set_training(False)
        return self.forward(x, batch_size)
    
    def set_training(self, training: bool):
        self.training = training
        self.dropout.set_training(training)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32, verbose: bool = True) -> Dict:
        
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
            
            # Mini-batch training
            for i in range(0, len(X_shuffled), batch_size):
                batch_x = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Training step
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
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32) -> Dict:
        
        from sklearn.metrics import f1_score, classification_report
        
        self.set_training(False)
        
        # Get predictions
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
        weights = {
            'embedding': self.embedding.get_parameters(),
            'lstm_layers': [],
            'dense': self.dense.get_parameters()
        }
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            if self.bidirectional:
                layer_weights = {
                    'forward': lstm_layer.forward_lstm.lstm_cell.get_parameters(),
                    'backward': lstm_layer.backward_lstm.lstm_cell.get_parameters()
                }
            else:
                layer_weights = lstm_layer.lstm_cell.get_parameters()
            
            weights['lstm_layers'].append(layer_weights)
        
        np.save(filepath, weights)
    
    def load_weights(self, filepath: str):
        """
        Load model weights from file
        """
        weights = np.load(filepath, allow_pickle=True).item()
        
        # Load embedding weights
        self.embedding.weights = weights['embedding']['weights']
        
        # Load LSTM weights
        for i, layer_weights in enumerate(weights['lstm_layers']):
            if self.bidirectional:
                # Load forward LSTM weights
                forward_weights = layer_weights['forward']
                for param_name, param_value in forward_weights.items():
                    setattr(self.lstm_layers[i].forward_lstm.lstm_cell, param_name, param_value)
                
                # Load backward LSTM weights
                backward_weights = layer_weights['backward']
                for param_name, param_value in backward_weights.items():
                    setattr(self.lstm_layers[i].backward_lstm.lstm_cell, param_name, param_value)
            else:
                # Load unidirectional LSTM weights
                for param_name, param_value in layer_weights.items():
                    setattr(self.lstm_layers[i].lstm_cell, param_name, param_value)
        
        # Load dense weights
        self.dense.W = weights['dense']['W']
        self.dense.b = weights['dense']['b']
    
    def load_keras_weights(self, keras_weights: Dict):
        
        if 'embedding' in keras_weights:
            self.embedding.weights = keras_weights['embedding']
        
        if 'lstm_layers' in keras_weights:
            for i, layer_weights in enumerate(keras_weights['lstm_layers']):
                if self.bidirectional:
                    if 'forward' in layer_weights:
                        forward_cell = self.lstm_layers[i].forward_lstm.lstm_cell
                        forward_cell.load_weights(layer_weights['forward'])
                    if 'backward' in layer_weights:
                        backward_cell = self.lstm_layers[i].backward_lstm.lstm_cell
                        backward_cell.load_weights(layer_weights['backward'])
                else:
                    self.lstm_layers[i].lstm_cell.load_weights(layer_weights)
        
        if 'dense' in keras_weights:
            self.dense.W = keras_weights['dense']['kernel']
            self.dense.b = keras_weights['dense']['bias']