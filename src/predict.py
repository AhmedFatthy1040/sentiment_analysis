import os
import argparse
import tensorflow as tf
import pickle

def load_model_and_tokenizer(model_path, tokenizer_path=None):
    """
    Load a trained model and tokenizer
    
    Args:
        model_path: Path to the saved model
        tokenizer_path: Path to the saved tokenizer (optional)
        
    Returns:
        model, tokenizer
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # If tokenizer path is not specified, look in the same directory as the model
    if tokenizer_path is None:
        tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer.pickle')
    
    # Load tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

def predict_sentiment(model, tokenizer, text, max_len=200):
    """
    Predict sentiment of a text
    
    Args:
        model: Trained sentiment analysis model
        tokenizer: Tokenizer used for training
        text: Text to analyze
        max_len: Maximum sequence length
    
    Returns:
        Sentiment prediction (0-1) where 1 is positive
    """
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    
    # Get prediction
    prediction = model.predict(padded)[0][0]
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    confidence = max(prediction, 1 - prediction) * 100
    
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw score: {prediction:.4f} (0=Negative, 1=Positive)")
    
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Make sentiment predictions using a trained model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model file')
    parser.add_argument('--tokenizer_path', type=str, 
                        help='Path to the saved tokenizer file (optional)')
    parser.add_argument('--text', type=str, 
                        help='Text to analyze')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--max_len', type=int, default=200,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    print("Model and tokenizer loaded successfully!")
    
    # Interactive mode
    if args.interactive:
        print("\nSentiment Analysis - Interactive Mode")
        print("Enter 'q' or 'quit' to exit")
        
        while True:
            text = input("\nEnter text to analyze: ")
            if text.lower() in ['q', 'quit', 'exit']:
                break
                
            predict_sentiment(model, tokenizer, text, max_len=args.max_len)
    
    # Single prediction mode
    elif args.text:
        predict_sentiment(model, tokenizer, args.text, max_len=args.max_len)
    
    # No text provided
    else:
        print("Error: Please provide text to analyze with --text or use --interactive mode")

if __name__ == '__main__':
    main()