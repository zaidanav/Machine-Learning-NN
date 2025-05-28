import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# Import our implementation
from rnn_implementation import SimpleRNNModel
from rnn_experiments import TextTokenization, preprocess_dataset

class RNNValidation:
    """
    Class to validate our RNN implementation against Keras
    """
    
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, 
                 num_rnn_layers=1, bidirectional=False):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.num_classes = num_classes
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        
        # Initialize both models
        self.keras_model = None
        self.scratch_model = None
        
    def create_keras_model(self):
        """Create equivalent Keras model"""
        model = keras.Sequential()
        
        # Embedding layer
        model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding'
        ))
        
        # RNN layers
        for i in range(self.num_rnn_layers):
            return_sequences = True if i < self.num_rnn_layers - 1 else False
            
            if self.bidirectional:
                rnn_layer = layers.Bidirectional(
                    layers.SimpleRNN(
                        self.rnn_units,
                        return_sequences=return_sequences,
                        name=f'rnn_{i}'
                    ),
                    name=f'bidirectional_rnn_{i}'
                )
            else:
                rnn_layer = layers.SimpleRNN(
                    self.rnn_units,
                    return_sequences=return_sequences,
                    name=f'rnn_{i}'
                )
            
            model.add(rnn_layer)
        
        # Dropout layer
        model.add(layers.Dropout(0.3, name='dropout'))
        
        # Dense layer
        model.add(layers.Dense(
            self.num_classes,
            activation='softmax',
            name='dense'
        ))
        
        # Build the model with proper input shape
        model.build(input_shape=(None, None))  # Build with variable sequence length
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.keras_model = model
        return model
    
    def create_scratch_model(self):
        """Create our from-scratch model"""
        self.scratch_model = SimpleRNNModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            rnn_units=self.rnn_units,
            num_classes=self.num_classes,
            num_rnn_layers=self.num_rnn_layers,
            bidirectional=self.bidirectional,
            dropout_rate=0.3,
            learning_rate=0.001
        )
        return self.scratch_model
    
    def extract_keras_weights(self):
        """Extract weights from trained Keras model"""
        if self.keras_model is None:
            raise ValueError("Keras model not created yet!")
        
        weights = {}
        
        # Extract embedding weights
        embedding_layer = self.keras_model.get_layer('embedding')
        weights['embedding'] = {
            'weights': embedding_layer.get_weights()[0]
        }
        
        # Extract RNN weights
        weights['rnn_layers'] = []
        
        for i in range(self.num_rnn_layers):
            if self.bidirectional:
                # Bidirectional RNN
                bi_layer = self.keras_model.get_layer(f'bidirectional_rnn_{i}')
                forward_weights, backward_weights = bi_layer.get_weights()
                
                # Keras bidirectional weights are structured differently
                # We need to split them properly
                keras_weights = bi_layer.get_weights()
                
                # Keras bidirectional RNN returns weights in a specific format
                # Handle different weight arrangements
                if len(keras_weights) == 6:  # Standard case: 2 sets of (W_ih, W_hh, b_h)
                    W_ih_f = keras_weights[0]  # Input to hidden forward
                    W_hh_f = keras_weights[1]  # Hidden to hidden forward
                    b_h_f = keras_weights[2]   # Bias forward
                    
                    W_ih_b = keras_weights[3]  # Input to hidden backward
                    W_hh_b = keras_weights[4]  # Hidden to hidden backward
                    b_h_b = keras_weights[5]   # Bias backward
                else:
                    # Handle alternative weight arrangement
                    # Sometimes Keras packs weights differently
                    forward_weights = keras_weights[:len(keras_weights)//2]
                    backward_weights = keras_weights[len(keras_weights)//2:]
                    
                    W_ih_f, W_hh_f, b_h_f = forward_weights[0], forward_weights[1], forward_weights[2] if len(forward_weights) >= 3 else np.zeros(self.rnn_units)
                    W_ih_b, W_hh_b, b_h_b = backward_weights[0], backward_weights[1], backward_weights[2] if len(backward_weights) >= 3 else np.zeros(self.rnn_units)
                
                layer_weights = {
                    'forward': {
                        'W_ih': W_ih_f,
                        'W_hh': W_hh_f,
                        'b_h': b_h_f
                    },
                    'backward': {
                        'W_ih': W_ih_b,
                        'W_hh': W_hh_b,
                        'b_h': b_h_b
                    }
                }
            else:
                # Unidirectional RNN
                rnn_layer = self.keras_model.get_layer(f'rnn_{i}')
                keras_weights = rnn_layer.get_weights()
                
                layer_weights = {
                    'W_ih': keras_weights[0],  # Input to hidden
                    'W_hh': keras_weights[1],  # Hidden to hidden
                    'b_h': keras_weights[2]    # Bias
                }
            
            weights['rnn_layers'].append(layer_weights)
        
        # Extract dense weights
        dense_layer = self.keras_model.get_layer('dense')
        dense_weights = dense_layer.get_weights()
        weights['dense'] = {
            'kernel': dense_weights[0],  # Weight matrix
            'bias': dense_weights[1]     # Bias vector
        }
        
        return weights
    
    def transfer_weights_to_scratch(self, keras_weights):
        """Transfer Keras weights to our scratch model"""
        if self.scratch_model is None:
            raise ValueError("Scratch model not created yet!")
        
        # Transfer embedding weights
        self.scratch_model.embedding.weights = keras_weights['embedding']['weights']
        
        # Transfer RNN weights
        for i, layer_weights in enumerate(keras_weights['rnn_layers']):
            if self.bidirectional:
                # Transfer bidirectional weights
                forward_cell = self.scratch_model.rnn_layers[i].forward_rnn.rnn_cell
                backward_cell = self.scratch_model.rnn_layers[i].backward_rnn.rnn_cell
                
                # Forward weights
                forward_cell.W_ih = layer_weights['forward']['W_ih']
                forward_cell.W_hh = layer_weights['forward']['W_hh']
                forward_cell.b_h = layer_weights['forward']['b_h']
                
                # Backward weights
                backward_cell.W_ih = layer_weights['backward']['W_ih']
                backward_cell.W_hh = layer_weights['backward']['W_hh']
                backward_cell.b_h = layer_weights['backward']['b_h']
            else:
                # Transfer unidirectional weights
                rnn_cell = self.scratch_model.rnn_layers[i].rnn_cell
                rnn_cell.W_ih = layer_weights['W_ih']
                rnn_cell.W_hh = layer_weights['W_hh']
                rnn_cell.b_h = layer_weights['b_h']
        
        # Transfer dense weights
        self.scratch_model.dense.W = keras_weights['dense']['kernel']
        self.scratch_model.dense.b = keras_weights['dense']['bias']
    
    def compare_predictions(self, X_test, y_test, sample_size=100):
        """Compare predictions between Keras and scratch models"""
        if self.keras_model is None or self.scratch_model is None:
            raise ValueError("Both models must be created first!")
        
        # Use a subset for detailed comparison
        X_sample = X_test[:sample_size]
        y_sample = y_test[:sample_size]
        
        # Get predictions from both models
        keras_pred_proba = self.keras_model.predict(X_sample, verbose=0)
        keras_pred = np.argmax(keras_pred_proba, axis=1)
        
        # Set scratch model to inference mode
        self.scratch_model.set_training(False)
        scratch_pred_proba = self.scratch_model.predict_proba(X_sample)
        scratch_pred = np.argmax(scratch_pred_proba, axis=1)
        
        # Calculate metrics
        keras_f1 = f1_score(y_sample, keras_pred, average='macro')
        scratch_f1 = f1_score(y_sample, scratch_pred, average='macro')
        
        keras_acc = np.mean(keras_pred == y_sample)
        scratch_acc = np.mean(scratch_pred == y_sample)
        
        # Calculate prediction similarity
        pred_similarity = np.mean(keras_pred == scratch_pred)
        prob_mse = np.mean((keras_pred_proba - scratch_pred_proba) ** 2)
        
        results = {
            'keras_f1': keras_f1,
            'scratch_f1': scratch_f1,
            'keras_accuracy': keras_acc,
            'scratch_accuracy': scratch_acc,
            'prediction_similarity': pred_similarity,
            'probability_mse': prob_mse,
            'keras_predictions': keras_pred,
            'scratch_predictions': scratch_pred,
            'keras_probabilities': keras_pred_proba,
            'scratch_probabilities': scratch_pred_proba
        }
        
        return results
    
    def validate_implementation(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Complete validation process"""
        print("Starting RNN Implementation Validation...")
        
        # Create both models
        print("1. Creating Keras model...")
        self.create_keras_model()
        print(f"   Keras model created with {self.keras_model.count_params()} parameters")
        
        print("2. Creating scratch model...")
        self.create_scratch_model()
        print("   Scratch model created")
        
        # Train Keras model
        print("3. Training Keras model...")
        history = self.keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=1
        )
        
        # Extract and transfer weights
        print("4. Extracting Keras weights...")
        keras_weights = self.extract_keras_weights()
        
        print("5. Transferring weights to scratch model...")
        self.transfer_weights_to_scratch(keras_weights)
        
        # Compare predictions
        print("6. Comparing predictions...")
        comparison_results = self.compare_predictions(X_test, y_test, sample_size=200)
        
        # Print results
        self.print_validation_results(comparison_results)
        
        # Create visualization
        self.create_validation_plots(comparison_results, history)
        
        return comparison_results
    
    def print_validation_results(self, results):
        """Print detailed validation results"""
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        print(f"Keras Model Performance:")
        print(f"  - Accuracy: {results['keras_accuracy']:.4f}")
        print(f"  - Macro F1: {results['keras_f1']:.4f}")
        
        print(f"\nScratch Model Performance:")
        print(f"  - Accuracy: {results['scratch_accuracy']:.4f}")
        print(f"  - Macro F1: {results['scratch_f1']:.4f}")
        
        print(f"\nModel Comparison:")
        print(f"  - Prediction Similarity: {results['prediction_similarity']:.4f}")
        print(f"  - Probability MSE: {results['probability_mse']:.6f}")
        
        # Performance difference
        acc_diff = abs(results['keras_accuracy'] - results['scratch_accuracy'])
        f1_diff = abs(results['keras_f1'] - results['scratch_f1'])
        
        print(f"\nPerformance Differences:")
        print(f"  - Accuracy Difference: {acc_diff:.4f}")
        print(f"  - F1 Score Difference: {f1_diff:.4f}")
        
        # Validation status
        if results['prediction_similarity'] > 0.95 and results['probability_mse'] < 0.01:
            print(f"\n VALIDATION SUCCESSFUL!")
            print(f"   Implementation matches Keras with high accuracy")
        elif results['prediction_similarity'] > 0.90 and results['probability_mse'] < 0.05:
            print(f"\n  VALIDATION MOSTLY SUCCESSFUL")
            print(f"   Implementation closely matches Keras with minor differences")
        else:
            print(f"\n VALIDATION NEEDS ATTENTION")
            print(f"   Significant differences detected between implementations")
    
    def create_validation_plots(self, results, history):
        """Create validation visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Training history
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Keras Model Training History')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        models = ['Keras', 'Scratch']
        accuracies = [results['keras_accuracy'], results['scratch_accuracy']]
        bars = axes[0, 1].bar(models, accuracies, alpha=0.7, color=['blue', 'orange'])
        axes[0, 1].set_title('Model Accuracy Comparison')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, axis='y', alpha=0.3)
        for bar, acc in zip(bars, accuracies):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{acc:.3f}', ha='center', va='bottom')
        
        # F1 score comparison
        f1_scores = [results['keras_f1'], results['scratch_f1']]
        bars = axes[0, 2].bar(models, f1_scores, alpha=0.7, color=['green', 'red'])
        axes[0, 2].set_title('Model F1-Score Comparison')
        axes[0, 2].set_ylabel('Macro F1-Score')
        axes[0, 2].grid(True, axis='y', alpha=0.3)
        for bar, f1 in zip(bars, f1_scores):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{f1:.3f}', ha='center', va='bottom')
        
        # Prediction scatter plot
        keras_preds = results['keras_predictions']
        scratch_preds = results['scratch_predictions']
        axes[1, 0].scatter(keras_preds, scratch_preds, alpha=0.6)
        axes[1, 0].plot([0, max(keras_preds.max(), scratch_preds.max())], 
                        [0, max(keras_preds.max(), scratch_preds.max())], 'r--')
        axes[1, 0].set_title('Prediction Comparison')
        axes[1, 0].set_xlabel('Keras Predictions')
        axes[1, 0].set_ylabel('Scratch Predictions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Probability comparison (first class)
        keras_probs = results['keras_probabilities'][:, 0]
        scratch_probs = results['scratch_probabilities'][:, 0]
        axes[1, 1].scatter(keras_probs, scratch_probs, alpha=0.6)
        axes[1, 1].plot([0, 1], [0, 1], 'r--')
        axes[1, 1].set_title('Probability Comparison (Class 0)')
        axes[1, 1].set_xlabel('Keras Probabilities')
        axes[1, 1].set_ylabel('Scratch Probabilities')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Validation metrics summary
        metrics = ['Pred Similarity', 'Prob MSE', 'Acc Diff', 'F1 Diff']
        values = [
            results['prediction_similarity'],
            results['probability_mse'] * 100,  # Scale for visibility
            abs(results['keras_accuracy'] - results['scratch_accuracy']),
            abs(results['keras_f1'] - results['scratch_f1'])
        ]
        
        bars = axes[1, 2].bar(metrics, values, alpha=0.7, color=['purple', 'brown', 'pink', 'gray'])
        axes[1, 2].set_title('Validation Metrics')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('rnn_validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_validation():
    """Run comprehensive validation of RNN implementation"""
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train = pd.read_csv('dataset/NusaX-Sentiment/train.csv')
    val = pd.read_csv('dataset/NusaX-Sentiment/valid.csv')
    test = pd.read_csv('dataset/NusaX-Sentiment/test.csv')
    
    # Initialize preprocessing
    tokenizer = TextTokenization(max_words=3000, max_sequence_length=50)
    encoder = LabelEncoder()
    
    # Preprocess data
    X_train, y_train = preprocess_dataset(train, tokenizer, encoder, is_train=True)
    X_val, y_val = preprocess_dataset(val, tokenizer, encoder, is_train=False)
    X_test, y_test = preprocess_dataset(test, tokenizer, encoder, is_train=False)
    
    print(f"Data preprocessed:")
    print(f"  - Vocabulary size: {tokenizer.vocab_size}")
    print(f"  - Number of classes: {len(encoder.classes_)}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Test samples: {len(X_test)}")
    
    # Test different configurations
    configurations = [
        {
            'name': 'Simple Unidirectional RNN',
            'rnn_units': 16,
            'num_layers': 1,
            'bidirectional': False
        },
        {
            'name': 'Deep Unidirectional RNN',
            'rnn_units': 16,
            'num_layers': 2,
            'bidirectional': False
        },
        {
            'name': 'Bidirectional RNN',
            'rnn_units': 16,
            'num_layers': 1,
            'bidirectional': True
        }
    ]
    
    validation_results = []
    
    for config in configurations:
        print(f"\n{'='*80}")
        print(f"VALIDATING: {config['name']}")
        print(f"{'='*80}")
        
        # Create validator
        validator = RNNValidation(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=32,
            rnn_units=config['rnn_units'],
            num_classes=len(encoder.classes_),
            num_rnn_layers=config['num_layers'],
            bidirectional=config['bidirectional']
        )
        
        # Run validation
        try:
            results = validator.validate_implementation(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            results['config'] = config
            validation_results.append(results)
            
        except Exception as e:
            print(f" Validation failed for {config['name']}: {str(e)}")
            continue
    
    # Print overall summary
    print_overall_validation_summary(validation_results)
    
    return validation_results


def print_overall_validation_summary(validation_results):
    """Print summary of all validation results"""
    
    print("\n" + "="*80)
    print("OVERALL VALIDATION SUMMARY")
    print("="*80)
    
    if not validation_results:
        print(" No successful validations completed!")
        return
    
    print(f" Successfully validated {len(validation_results)} configurations")
    print()
    
    for i, result in enumerate(validation_results, 1):
        config = result['config']
        print(f"{i}. {config['name']}:")
        print(f"   - Prediction Similarity: {result['prediction_similarity']:.4f}")
        print(f"   - Probability MSE: {result['probability_mse']:.6f}")
        print(f"   - Accuracy Difference: {abs(result['keras_accuracy'] - result['scratch_accuracy']):.4f}")
        print(f"   - F1 Score Difference: {abs(result['keras_f1'] - result['scratch_f1']):.4f}")
        
        # Validation status
        if result['prediction_similarity'] > 0.95 and result['probability_mse'] < 0.01:
            status = " EXCELLENT"
        elif result['prediction_similarity'] > 0.90 and result['probability_mse'] < 0.05:
            status = "GOOD"
        else:
            status = "NEEDS WORK"
        
        print(f"   - Status: {status}")
        print()
    
    # Best performing configuration
    best_config = max(validation_results, key=lambda x: x['prediction_similarity'])
    print(f"Best Performing Configuration:")
    print(f"   {best_config['config']['name']}")
    print(f"   Prediction Similarity: {best_config['prediction_similarity']:.4f}")
    
    print("\nImplementation Quality Assessment:")
    avg_similarity = np.mean([r['prediction_similarity'] for r in validation_results])
    avg_mse = np.mean([r['probability_mse'] for r in validation_results])
    
    if avg_similarity > 0.95 and avg_mse < 0.01:
        print("   IMPLEMENTATION EXCELLENT - Ready for production!")
    elif avg_similarity > 0.90 and avg_mse < 0.05:
        print("   IMPLEMENTATION GOOD - Minor optimizations possible")
    else:
        print("   IMPLEMENTATION NEEDS REFINEMENT")
    
    print(f"   Average Prediction Similarity: {avg_similarity:.4f}")
    print(f"   Average Probability MSE: {avg_mse:.6f}")


# Main execution
if __name__ == "__main__":
    print("RNN Implementation Validation")
    print("=" * 50)
    print("This script validates our RNN implementation against Keras")
    print("by comparing predictions and performance metrics.")
    print()
    
    validation_results = run_comprehensive_validation()
    
    print("\nValidation completed!")
    print("Check the generated plots and summary for detailed results.")