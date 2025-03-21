import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Local imports
from model import create_rnn_model, get_callbacks
from data_loader import load_imdb_dataset, load_custom_dataset

def plot_history(history):
    """
    Plot training & validation accuracy and loss values
    
    Args:
        history: History object returned from model.fit()
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def train_model(args):
    """
    Train the sentiment analysis model
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'model.h5')
    
    # Load data
    if args.custom_dataset:
        X_train, X_test, y_train, y_test, tokenizer = load_custom_dataset(
            args.custom_dataset,
            max_features=args.vocab_size,
            max_len=args.max_len,
            test_size=args.test_size
        )
    else:
        X_train, X_test, y_train, y_test, tokenizer = load_imdb_dataset(
            max_features=args.vocab_size,
            max_len=args.max_len,
            test_size=args.test_size
        )
    
    # Create model
    model = create_rnn_model(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        input_length=args.max_len,
        rnn_type=args.rnn_type
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = get_callbacks(checkpoint_path=checkpoint_path)
    
    # Add TensorBoard callback
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    
    # Train model
    print(f"\nTraining {args.rnn_type.upper()} model...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_history(history)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save the tokenizer
    import pickle
    with open(os.path.join(args.output_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nModel and tokenizer saved to {args.output_dir}")
    return model, tokenizer

def predict_sentiment(model, tokenizer, text, max_len=100):
    """
    Predict sentiment of a text
    
    Args:
        model: Trained sentiment analysis model
        tokenizer: Tokenizer used for training
        text: Text to analyze
        max_len: Maximum sequence length
    
    Returns:
        Sentiment score (0-1) where 1 is positive
    """
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    
    # Get prediction
    prediction = model.predict(padded)[0][0]
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment} (score: {prediction:.4f})")
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model with RNN')
    
    # Dataset parameters
    parser.add_argument('--custom_dataset', type=str, help='Path to custom dataset CSV file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of vocabulary')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], 
                        help='Type of RNN cell to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Prediction mode
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--model_path', type=str, help='Path to saved model for prediction')
    parser.add_argument('--text', type=str, help='Text for sentiment prediction')
    
    args = parser.parse_args()
    
    # Prediction mode
    if args.predict:
        if not args.model_path:
            print("Error: --model_path is required for prediction mode")
            return
            
        if not args.text:
            print("Error: --text is required for prediction mode")
            return
            
        # Load model and tokenizer
        model = tf.keras.models.load_model(args.model_path)
        
        with open(os.path.join(os.path.dirname(args.model_path), 'tokenizer.pickle'), 'rb') as handle:
            import pickle
            tokenizer = pickle.load(handle)
            
        # Make prediction
        predict_sentiment(model, tokenizer, args.text, max_len=args.max_len)
        
    else:
        # Training mode
        train_model(args)
        
if __name__ == '__main__':
    main()