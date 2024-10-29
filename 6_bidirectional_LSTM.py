# Implement bidirectional LSTM for sentiment analysis on movie reviews

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the IMDB dataset
max_words = 10000  # Limit the number of words
max_len = 200      # Maximum length of each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Pad sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# Define the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=False)))  # Bidirectional LSTM layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

def predict_review(review):
    # Preprocess the review (this should be done similarly to the training data)
    # For demonstration, assume `review` is already tokenized and padded.
    review = pad_sequences([review], maxlen=max_len)
    prediction = model.predict(review)
    return 'Positive' if prediction[0][0] > 0.5 else 'Negative'

# Example usage
# review = ... # Your tokenized review
# print(predict_review(review))