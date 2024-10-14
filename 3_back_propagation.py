# Build a deep feed forward ANN by implementing the backpropagation algorithm an test the same using approopriate dataset. Use the hidden layers >=4

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Helper Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = - np.log(y_pred[range(n_samples), y_true.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

# Initialize Weights
def initialize_weights(input_size, hidden_sizes, output_size):
    weights = []
    biases = []
    
    # Input to first hidden layer
    weights.append(np.random.randn(input_size, hidden_sizes[0]))
    biases.append(np.zeros((1, hidden_sizes[0])))
    
    # Hidden layers
    for i in range(1, len(hidden_sizes)):
        weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]))
        biases.append(np.zeros((1, hidden_sizes[i])))
    
    # Hidden to output layer
    weights.append(np.random.randn(hidden_sizes[-1], output_size))
    biases.append(np.zeros((1, output_size)))
    
    return weights, biases

# Feedforward
def feedforward(X, weights, biases):
    layers = [X]
    for i in range(len(weights) - 1):
        Z = np.dot(layers[-1], weights[i]) + biases[i]
        A = sigmoid(Z)
        layers.append(A)
    
    Z_out = np.dot(layers[-1], weights[-1]) + biases[-1]
    A_out = softmax(Z_out)
    layers.append(A_out)
    
    return layers

# Backpropagation
def backpropagation(y_true, weights, biases, layers, learning_rate):
    y_pred = layers[-1]
    deltas = [cross_entropy_derivative(y_true, y_pred)]
    
    # Backpropagation through hidden layers
    for i in reversed(range(len(weights) - 1)):
        delta = np.dot(deltas[-1], weights[i + 1].T) * sigmoid_derivative(layers[i + 1])
        deltas.append(delta)
    
    deltas.reverse()
    
    # Update weights and biases
    for i in range(len(weights)):
        weights[i] -= learning_rate * np.dot(layers[i].T, deltas[i])
        biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

# Train Function
def train(X, y, input_size, hidden_sizes, output_size, epochs, learning_rate):
    weights, biases = initialize_weights(input_size, hidden_sizes, output_size)
    
    for epoch in range(epochs):
        layers = feedforward(X, weights, biases)
        loss = cross_entropy_loss(y, layers[-1])
        backpropagation(y, weights, biases, layers, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss}")
    
    return weights, biases

# Predict Function
def predict(X, weights, biases):
    layers = feedforward(X, weights, biases)
    return np.argmax(layers[-1], axis=1)

# Load and Prepare Dataset (MNIST)
def load_mnist():
    mnist = fetch_openml('mnist_784')
    X = mnist.data
    y = mnist.target.astype(int)
    
    # Normalize
    X = X / 255.0
    
    # One hot encode labels
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Main Program
if __name__ == '__main__':
    # Load data
    X_train, X_test, y_train, y_test = load_mnist()
    
    # Parameters
    input_size = 784  # 28x28 pixels
    hidden_sizes = [128, 64, 64, 32]  # 4 hidden layers
    output_size = 10  # 10 classes (digits 0-9)
    epochs = 1000
    learning_rate = 0.01
    
    # Train model
    weights, biases = train(X_train, y_train, input_size, hidden_sizes, output_size, epochs, learning_rate)
    
    # Test model
    y_pred = predict(X_test, weights, biases)
    y_test_labels = np.argmax(y_test, axis=1)
    
    accuracy = np.mean(y_pred == y_test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
