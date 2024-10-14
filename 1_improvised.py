# Write a program to demonstrate the working of different activation functions like Sigmoid, Tanh, RELU and softmax to train neural network.

import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max to prevent overflow
    return exp_x / exp_x.sum(axis=0)

# Create input values
x = np.linspace(-10, 10, 400)

# Apply activation functions
sigmoid_values = sigmoid(x)
tanh_values = tanh(x)
relu_values = relu(x)

# For softmax, we use a set of values (instead of point-wise) to illustrate distribution
softmax_input = np.array([x, x/2, x/3])
softmax_values = softmax(softmax_input)

# Plotting activation functions
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_values, label="Sigmoid")
plt.title("Sigmoid Function")
plt.grid(True)

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x, tanh_values, label="Tanh", color='orange')
plt.title("Tanh Function")
plt.grid(True)

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x, relu_values, label="ReLU", color='green')
plt.title("ReLU Function")
plt.grid(True)

# Softmax
plt.subplot(2, 2, 4)
plt.plot(x, softmax_values[0], label="Softmax - Class 1", color='red')
plt.plot(x, softmax_values[1], label="Softmax - Class 2", color='blue')
plt.plot(x, softmax_values[2], label="Softmax - Class 3", color='purple')
plt.title("Softmax Function")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()