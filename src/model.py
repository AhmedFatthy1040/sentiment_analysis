import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_rnn_model(vocab_size, embedding_dim=100, input_length=100, rnn_type='lstm'):
    """
    Create a recurrent neural network model for sentiment analysis.
    
    Args:
        vocab_size: Size of the vocabulary (number of unique words)
        embedding_dim: Dimension of the embedding layer
        input_length: Maximum length of input sequences
        rnn_type: Type of RNN cell ('lstm' or 'gru')
    
    Returns:
        A compiled Keras model
    """
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    
    # RNN layer
    if rnn_type.lower() == 'lstm':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(32)))
    elif rnn_type.lower() == 'gru':
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Bidirectional(GRU(32)))
    else:
        raise ValueError(f"Unsupported RNN type: {rnn_type}")
    
    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def get_callbacks(checkpoint_path='model_checkpoint.h5'):
    """
    Get callbacks for training the model.
    
    Args:
        checkpoint_path: Path to save the best model
        
    Returns:
        List of callbacks
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    return [early_stopping, model_checkpoint]