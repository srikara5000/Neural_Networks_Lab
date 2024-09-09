import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)  # One-hot encoding

activation_functions = ['sigmoid', 'tanh', 'relu', 'softmax']

# Dictionary to hold the training histories for each activation function
histories = {}

for activation in activation_functions:
    # Define the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into vectors of size 784
        Dense(128, activation=activation),  # Hidden layer with 128 units and the current activation function
        Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit)
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    print(f"\nTraining model with {activation} activation function:")
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)
    
    # Store the training history
    histories[activation] = history

# Plot the training results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, activation in enumerate(activation_functions):
    ax = axes[i]
    ax.plot(histories[activation].history['accuracy'], label='Training Accuracy')
    ax.plot(histories[activation].history['val_accuracy'], label='Validation Accuracy')
    ax.set_title(f'Activation Function: {activation}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

plt.tight_layout()
plt.show()