import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_imdb_dataset(max_features=10000, max_len=100, test_size=0.2, random_state=42):
    """
    Load the IMDB dataset for sentiment analysis.
    
    Args:
        max_features: Maximum number of words to consider in the vocabulary
        max_len: Maximum length of sequences
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, tokenizer
    """
    # Download the IMDB dataset
    print("Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features
    )
    
    # Get word index mapping
    word_index = tf.keras.datasets.imdb.get_word_index()
    
    # Create a reverse mapping
    reverse_word_index = {i+3: word for word, i in word_index.items()}
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'
    
    # Convert integer sequences back to text
    def decode_review(encoded_text):
        return ' '.join([reverse_word_index.get(i, '?') for i in encoded_text])
    
    # Convert all reviews back to text
    print("Converting reviews to text...")
    x_train_text = [decode_review(x) for x in x_train]
    x_test_text = [decode_review(x) for x in x_test]
    
    # Tokenize the text again to ensure consistency
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train_text + x_test_text)
    
    X_train = tokenizer.texts_to_sequences(x_train_text)
    X_test = tokenizer.texts_to_sequences(x_test_text)
    
    # Pad sequences to ensure uniform input length
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, tokenizer

def load_custom_dataset(data_path, max_features=10000, max_len=100, test_size=0.2, random_state=42):
    """
    Load a custom dataset for sentiment analysis.
    
    Args:
        data_path: Path to the CSV file containing 'text' and 'sentiment' columns
        max_features: Maximum number of words to consider in the vocabulary
        max_len: Maximum length of sequences
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, tokenizer
    """
    import pandas as pd
    
    print(f"Loading custom dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure the dataframe has the required columns
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'sentiment' columns")
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Tokenize the text
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df['text'].values)
    
    X_train = tokenizer.texts_to_sequences(train_df['text'].values)
    X_test = tokenizer.texts_to_sequences(test_df['text'].values)
    
    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    # Convert labels
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, tokenizer