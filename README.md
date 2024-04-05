# Sentiment Analysis Using Pretrained Embeddings and RNN
This project demonstrates training custom Word2Vec and FastText embeddings on text data, followed by utilizing these embeddings in a Recurrent Neural Network (RNN) model for the task of sentiment analysis. It highlights the process of saving and loading the trained embeddings and the RNN model for reuse.

# Getting Started
- Prerequisites
- Python 3.x
- Gensim
- TensorFlow
- NLTK
- Pandas
- Installation

## Project Structure

Word2Vec Model Training: Training a Word2Vec model using Skip-Gram and Negative Sampling.

FastText Model Training: Training a FastText model to capture subword information.

RNN Model Building and Testing: Using pretrained embeddings to train an RNN model and evaluate its performance on sentiment classification.

### Word2Vec and FastText Model Training
Both models are trained on a dataset containing text data. The data is preprocessed to remove stopwords and punctuation, and then tokenized. Each model is saved to disk after training for later use in the RNN model.

- Data Preparation: Your dataset should be in a CSV format with a column named 'text'.
- Training: The script word2vec_training.py and fasttext_training.py contain the code for training each model respectively.
- Model Saving: Models are saved as word2vec.model and fasttext.model.

### RNN Model Building and Testing
The RNN model utilizes the pretrained embeddings from the Word2Vec or FastText models. The model is then evaluated on a separate test dataset to determine its loss and accuracy.

- Embedding Matrix Preparation: Converts the pretrained embeddings into a matrix.
- RNN Architecture: The model uses a bidirectional LSTM layer.
- Evaluation: The test dataset is processed similarly to the training dataset, and the model's performance is measured.

