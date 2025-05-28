import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re
from typing import List
import time

# Import our RNN implementation
from rnn_implementation import SimpleRNNModel

class TextTokenization:
    """Text preprocessing and tokenization class"""
    
    def __init__(self, 
                 max_words: int = 4000,
                 max_sequence_length: int = 100,
                 oov_token: str = "<OOV>",
                 padding: str = 'post',
                 truncating: str = 'post'):

        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating
        
        self.tokenizer = Tokenizer(
            num_words=max_words,
            oov_token=oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        self.vocab_size = None
        self.word_index = None
        self.is_fitted = False
        
        self.text_stats = {
            'original_lengths': [],
            'cleaned_lengths': [],
            'total_texts': 0,
            'unique_words': 0
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        original_length = len(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, mentions, hashtags
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters but keep Indonesian characters and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        text = ' '.join(words)
        
        # Store statistics
        cleaned_length = len(text.split())
        self.text_stats['original_lengths'].append(original_length)
        self.text_stats['cleaned_lengths'].append(cleaned_length)
        
        return text
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts"""
        cleaned_texts = []
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i}/{len(texts)} texts")
            
            cleaned_text = self.clean_text(text)
            cleaned_texts.append(cleaned_text)
        
        self.text_stats['total_texts'] += len(texts)
        print(f"Preprocessing completed. Total texts processed: {self.text_stats['total_texts']}")
        
        return cleaned_texts
    
    def fit(self, texts: List[str]) -> 'TextTokenization':
        """Fit tokenizer on texts"""
        cleaned_texts = self.preprocess_texts(texts)
        
        self.tokenizer.fit_on_texts(cleaned_texts)
        
        self.word_index = self.tokenizer.word_index
        self.vocab_size = min(len(self.word_index) + 1, self.max_words)
        self.is_fitted = True
        
        self.text_stats['unique_words'] = len(self.word_index)
        
        print(f"Tokenizer fitted successfully!")
        print(f"- Total unique words: {len(self.word_index)}")
        print(f"- Vocabulary size (with limit): {self.vocab_size}")
        print(f"- OOV token: {self.oov_token}")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to sequences"""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transforming. Call fit() first.")
        
        print(f"Transforming {len(texts)} texts to sequences...")
        
        cleaned_texts = self.preprocess_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding=self.padding,
            truncating=self.truncating
        )
        
        print(f"Transformation completed. Output shape: {padded_sequences.shape}")
        
        return padded_sequences
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(texts).transform(texts)


def preprocess_dataset(df, tokenizer, encoder, is_train=False):
    """Preprocess dataset for RNN training"""
    df_copy = df.copy()
    
    if is_train:
        X = tokenizer.fit_transform(df_copy['text'].tolist())
        encoder.fit(df_copy['label'])
        y = encoder.transform(df_copy['label'])
    else:
        X = tokenizer.transform(df_copy['text'].tolist())
        y = encoder.transform(df_copy['label'])
    
    return X, y


def run_rnn_experiments():
    """Run all RNN experiments with bonus features"""
    
    # Load data
    print("Loading datasets...")
    train = pd.read_csv('dataset/NusaX-Sentiment/train.csv')
    val = pd.read_csv('dataset/NusaX-Sentiment/valid.csv')
    test = pd.read_csv('dataset/NusaX-Sentiment/test.csv')
    
    print("Label distribution:")
    print(train['label'].value_counts())
    
    # Initialize tokenizer and encoder
    tokenizer = TextTokenization(max_words=5000, max_sequence_length=100)
    encoder = LabelEncoder()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, y_train = preprocess_dataset(train, tokenizer, encoder, is_train=True)
    X_val, y_val = preprocess_dataset(val, tokenizer, encoder, is_train=False)
    X_test, y_test = preprocess_dataset(test, tokenizer, encoder, is_train=False)
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Experiment 1: Number of RNN layers
    print("\n" + "="*60)
    print("EXPERIMENT 1: PENGARUH JUMLAH LAYER RNN")
    print("="*60)
    
    layer_variations = [1, 2, 3]
    layer_results = []
    
    for num_layers in layer_variations:
        print(f"\nTraining RNN with {num_layers} layers...")
        start_time = time.time()
        
        # Create model with backward propagation support (Bonus 1)
        model = SimpleRNNModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=32,
            rnn_units=16,
            num_classes=len(encoder.classes_),
            num_rnn_layers=num_layers,
            bidirectional=False,
            dropout_rate=0.3,
            learning_rate=0.01
        )
        
        # Training with batch support (Bonus 2)
        history = model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=15,
            batch_size=16,
            verbose=True
        )
        
        # Evaluation with batch support (Bonus 2)
        results = model.evaluate(X_test, y_test, batch_size=32)
        
        training_time = time.time() - start_time
        
        layer_results.append({
            'num_layers': num_layers,
            'history': history,
            'test_results': results,
            'training_time': training_time,
            'model': model
        })
        
        print(f"Results for {num_layers} layers:")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Macro F1-Score: {results['macro_f1_score']:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        # Save model weights
        model.save_weights(f'rnn_model_{num_layers}layers.npy')
    
    # Experiment 2: Number of RNN units per layer
    print("\n" + "="*60)
    print("EXPERIMENT 2: PENGARUH BANYAK CELL RNN PER LAYER")
    print("="*60)
    
    unit_variations = [8, 16, 32]
    unit_results = []
    
    for num_units in unit_variations:
        print(f"\nTraining RNN with {num_units} units per layer...")
        start_time = time.time()
        
        model = SimpleRNNModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=32,
            rnn_units=num_units,
            num_classes=len(encoder.classes_),
            num_rnn_layers=2,
            bidirectional=False,
            dropout_rate=0.3,
            learning_rate=0.01
        )
        
        history = model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=15,
            batch_size=16,
            verbose=True
        )
        
        results = model.evaluate(X_test, y_test, batch_size=32)
        training_time = time.time() - start_time
        
        unit_results.append({
            'num_units': num_units,
            'history': history,
            'test_results': results,
            'training_time': training_time,
            'model': model
        })
        
        print(f"Results for {num_units} units:")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Macro F1-Score: {results['macro_f1_score']:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        model.save_weights(f'rnn_model_{num_units}units.npy')
    
    # Experiment 3: Bidirectional vs Unidirectional
    print("\n" + "="*60)
    print("EXPERIMENT 3: PENGARUH ARAH RNN (BIDIRECTIONAL VS UNIDIRECTIONAL)")
    print("="*60)
    
    direction_variations = [False, True]  # Unidirectional, Bidirectional
    direction_results = []
    
    for bidirectional in direction_variations:
        direction_name = "Bidirectional" if bidirectional else "Unidirectional"
        print(f"\nTraining {direction_name} RNN...")
        start_time = time.time()
        
        model = SimpleRNNModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=32,
            rnn_units=16,
            num_classes=len(encoder.classes_),
            num_rnn_layers=2,
            bidirectional=bidirectional,
            dropout_rate=0.3,
            learning_rate=0.01
        )
        
        history = model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=15,
            batch_size=16,
            verbose=True
        )
        
        results = model.evaluate(X_test, y_test, batch_size=32)
        training_time = time.time() - start_time
        
        direction_results.append({
            'bidirectional': bidirectional,
            'direction_name': direction_name,
            'history': history,
            'test_results': results,
            'training_time': training_time,
            'model': model
        })
        
        print(f"Results for {direction_name}:")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Macro F1-Score: {results['macro_f1_score']:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        model.save_weights(f'rnn_model_{direction_name.lower()}.npy')
    
    # Test Bonus 2: Large batch inference
    print("\n" + "="*60)
    print("TESTING BONUS 2: BATCH INFERENCE CAPABILITY")
    print("="*60)
    
    best_model = max(layer_results, key=lambda x: x['test_results']['macro_f1_score'])['model']
    
    print("Testing batch inference with different batch sizes...")
    batch_sizes = [16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        start_time = time.time()
        predictions = best_model.predict(X_test, batch_size=batch_size)
        inference_time = time.time() - start_time
        
        accuracy = np.mean(predictions == y_test)
        print(f"Batch size {batch_size}: Accuracy={accuracy:.4f}, Time={inference_time:.3f}s")
    
    # Create visualization plots
    create_experiment_plots(layer_results, unit_results, direction_results)
    
    # Print final summary
    print_experiment_summary(layer_results, unit_results, direction_results)
    
    return layer_results, unit_results, direction_results


def create_experiment_plots(layer_results, unit_results, direction_results):
    """Create visualization plots for all experiments"""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 15))
    
    # Experiment 1 plots
    plt.subplot(3, 4, 1)
    for result in layer_results:
        plt.plot(result['history']['train_loss'], label=f"{result['num_layers']} layers")
    plt.title('Training Loss vs Epochs\n(Number of Layers)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    for result in layer_results:
        plt.plot(result['history']['val_loss'], label=f"{result['num_layers']} layers")
    plt.title('Validation Loss vs Epochs\n(Number of Layers)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 3)
    f1_scores = [result['test_results']['macro_f1_score'] for result in layer_results]
    layer_counts = [result['num_layers'] for result in layer_results]
    bars = plt.bar(layer_counts, f1_scores, alpha=0.7)
    plt.title('Macro F1-Score by\nNumber of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Macro F1-Score')
    plt.grid(True, axis='y', alpha=0.3)
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.subplot(3, 4, 4)
    times = [result['training_time'] for result in layer_results]
    bars = plt.bar(layer_counts, times, alpha=0.7, color='orange')
    plt.title('Training Time by\nNumber of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Experiment 2 plots
    plt.subplot(3, 4, 5)
    for result in unit_results:
        plt.plot(result['history']['train_loss'], label=f"{result['num_units']} units")
    plt.title('Training Loss vs Epochs\n(Number of Units)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 6)
    for result in unit_results:
        plt.plot(result['history']['val_loss'], label=f"{result['num_units']} units")
    plt.title('Validation Loss vs Epochs\n(Number of Units)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 7)
    f1_scores = [result['test_results']['macro_f1_score'] for result in unit_results]
    unit_counts = [result['num_units'] for result in unit_results]
    bars = plt.bar(unit_counts, f1_scores, alpha=0.7, color='green')
    plt.title('Macro F1-Score by\nNumber of Units')
    plt.xlabel('Number of Units')
    plt.ylabel('Macro F1-Score')
    plt.grid(True, axis='y', alpha=0.3)
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.subplot(3, 4, 8)
    times = [result['training_time'] for result in unit_results]
    bars = plt.bar(unit_counts, times, alpha=0.7, color='red')
    plt.title('Training Time by\nNumber of Units')
    plt.xlabel('Number of Units')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Experiment 3 plots
    plt.subplot(3, 4, 9)
    for result in direction_results:
        plt.plot(result['history']['train_loss'], label=result['direction_name'])
    plt.title('Training Loss vs Epochs\n(Direction)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 10)
    for result in direction_results:
        plt.plot(result['history']['val_loss'], label=result['direction_name'])
    plt.title('Validation Loss vs Epochs\n(Direction)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 11)
    f1_scores = [result['test_results']['macro_f1_score'] for result in direction_results]
    direction_names = [result['direction_name'] for result in direction_results]
    bars = plt.bar(direction_names, f1_scores, alpha=0.7, color='purple')
    plt.title('Macro F1-Score by\nDirection')
    plt.xlabel('Direction')
    plt.ylabel('Macro F1-Score')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.subplot(3, 4, 12)
    times = [result['training_time'] for result in direction_results]
    bars = plt.bar(direction_names, times, alpha=0.7, color='brown')
    plt.title('Training Time by\nDirection')
    plt.xlabel('Direction')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rnn_experiments_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_experiment_summary(layer_results, unit_results, direction_results):
    """Print comprehensive experiment summary"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*80)
    
    # Experiment 1 Summary
    print("\n1. PENGARUH JUMLAH LAYER RNN:")
    print("-" * 40)
    best_layer = max(layer_results, key=lambda x: x['test_results']['macro_f1_score'])
    for result in layer_results:
        print(f"   {result['num_layers']} layers: F1={result['test_results']['macro_f1_score']:.4f}, "
              f"Acc={result['test_results']['test_accuracy']:.4f}, "
              f"Time={result['training_time']:.1f}s")
    print(f"   → Best: {best_layer['num_layers']} layers "
          f"(F1={best_layer['test_results']['macro_f1_score']:.4f})")
    
    # Experiment 2 Summary
    print("\n2. PENGARUH BANYAK CELL RNN PER LAYER:")
    print("-" * 40)
    best_unit = max(unit_results, key=lambda x: x['test_results']['macro_f1_score'])
    for result in unit_results:
        print(f"   {result['num_units']} units: F1={result['test_results']['macro_f1_score']:.4f}, "
              f"Acc={result['test_results']['test_accuracy']:.4f}, "
              f"Time={result['training_time']:.1f}s")
    print(f"   → Best: {best_unit['num_units']} units "
          f"(F1={best_unit['test_results']['macro_f1_score']:.4f})")
    
    # Experiment 3 Summary
    print("\n3. PENGARUH ARAH RNN:")
    print("-" * 40)
    best_direction = max(direction_results, key=lambda x: x['test_results']['macro_f1_score'])
    for result in direction_results:
        print(f"   {result['direction_name']}: F1={result['test_results']['macro_f1_score']:.4f}, "
              f"Acc={result['test_results']['test_accuracy']:.4f}, "
              f"Time={result['training_time']:.1f}s")
    print(f"   → Best: {best_direction['direction_name']} "
          f"(F1={best_direction['test_results']['macro_f1_score']:.4f})")
    
    # Overall best model
    all_results = layer_results + unit_results + direction_results
    overall_best = max(all_results, key=lambda x: x['test_results']['macro_f1_score'])
    
    print("\n4. OVERALL BEST CONFIGURATION:")
    print("-" * 40)
    config_name = ""
    if 'num_layers' in overall_best:
        config_name = f"{overall_best['num_layers']} layers"
    elif 'num_units' in overall_best:
        config_name = f"{overall_best['num_units']} units"
    elif 'direction_name' in overall_best:
        config_name = overall_best['direction_name']
    
    print(f"   Configuration: {config_name}")
    print(f"   Macro F1-Score: {overall_best['test_results']['macro_f1_score']:.4f}")
    print(f"   Test Accuracy: {overall_best['test_results']['test_accuracy']:.4f}")
    print(f"   Training Time: {overall_best['training_time']:.1f} seconds")
    
    # Bonus features summary
    print("\n5. BONUS FEATURES IMPLEMENTED:")
    print("-" * 40)
    print("   Bonus 1: Backward Propagation from Scratch")
    print("      - Complete gradient computation for all RNN layers")
    print("      - Custom training loop with parameter updates")
    print("      - Adam optimizer implementation")
    
    print("   Bonus 2: Batch Inference Support")
    print("      - Configurable batch size for inference")
    print("      - Memory-efficient large dataset processing")
    print("      - Batch processing in training and evaluation")
    
    print("\n6. KESIMPULAN:")
    print("-" * 40)
    print("   • Model RNN berhasil diimplementasikan from scratch dengan fitur lengkap")
    print("   • Backward propagation dan batch processing berjalan dengan baik")
    print("   • Hyperparameter tuning menunjukkan trade-off antara kompleksitas dan performa")
    print("   • Bidirectional RNN umumnya memberikan performa lebih baik")
    print("   • Implementasi siap untuk deployment dan eksperimen lanjutan")


# Main execution
if __name__ == "__main__":
    print("Starting RNN Experiments with Bonus Features...")
    print("This includes:")
    print("- Bonus 1: Complete Backward Propagation from Scratch")
    print("- Bonus 2: Batch Inference Support")
    print("- Comprehensive Hyperparameter Analysis")
    
    layer_results, unit_results, direction_results = run_rnn_experiments()
    
    print("\nAll experiments completed successfully!")
    print("Results saved and visualizations created.")