# Sentiment Analysis with RNN

This project implements sentiment analysis using Recurrent Neural Networks (RNN) such as LSTM and GRU. It can be used to classify text as having positive or negative sentiment.

## Project Structure

- `model.py`: Defines the RNN model architecture
- `data_loader.py`: Functions to load and preprocess sentiment datasets
- `train.py`: Script for training the sentiment analysis model
- `predict.py`: Script for making predictions with a trained model
- `requirements.txt`: Required packages

## Setup

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Training the Model

You can train the model using the built-in IMDB dataset or a custom dataset.

### Using the IMDB Dataset

```bash
python train.py --rnn_type lstm --epochs 5 --output_dir ./output
```

### Using a Custom Dataset

Your custom dataset should be a CSV file with at least two columns: 'text' (containing the text to analyze) and 'sentiment' (containing 0 for negative sentiment and 1 for positive sentiment).

```bash
python train.py --custom_dataset path/to/your/dataset.csv --rnn_type lstm --epochs 5 --output_dir ./output
```

### Training Options

- `--rnn_type`: Type of RNN cell to use ('lstm' or 'gru')
- `--vocab_size`: Maximum vocabulary size (default: 10000)
- `--embedding_dim`: Dimension of word embeddings (default: 100)
- `--max_len`: Maximum sequence length (default: 200)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 64)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--output_dir`: Directory to save model and tokenizer (default: './output')

## Making Predictions

After training, you can use the trained model to make predictions:

### Single Prediction

```bash
python predict.py --model_path ./output/model.h5 --text "This movie was amazing!"
```

### Interactive Mode

```bash
python predict.py --model_path ./output/model.h5 --interactive
```

## Model Architecture

This project uses a bidirectional RNN (LSTM or GRU) architecture:

1. Embedding layer: Converts words to dense vectors
2. Bidirectional LSTM/GRU layers: Capture sequential patterns in text
3. Dense layers: Map features to sentiment prediction
4. Sigmoid activation: Outputs probability of positive sentiment

## Performance

With the IMDB dataset, this model typically achieves around 85-90% accuracy after 5-10 epochs.

## Customization

You can customize the model architecture in `model.py` to experiment with different layer configurations, embedding dimensions, or additional techniques like attention mechanisms.

## License

This project is open source and available under the MIT License.