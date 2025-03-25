# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the IMDB dataset
from tensorflow.keras.datasets import imdb

# Hyperparameters
vocab_size = 10000  # Number of words in vocabulary
max_length = 200    # Max words per review
embedding_dim = 50  # Size of word embeddings
rnn_units = 64      # Number of RNN units (Hyperparameter)
dropout_rate = 0.3  # Dropout rate (Hyperparameter)
learning_rate = 0.001  # Learning rate for optimizer

# Load dataset (Only keep top 10,000 words)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding sequences to ensure equal length
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# Splitting data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),  # Word Embeddings
    SimpleRNN(rnn_units, activation='tanh', return_sequences=False),  # RNN Layer
    Dropout(dropout_rate),  # Dropout for regularization
    Dense(1, activation='sigmoid')  # Output Layer for binary classification
])

# Compile the model
optimizer = Adam(learning_rate=learning_rate)  # Adam optimizer with tuned learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=32)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
