# Build a deep feed forward ANN by implementing the backpropagation algorithm an test the same using appropriate dataset. Use the hidden layers >=4

import numpy as np

class DeepFeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.weights)):
            self.z.append(np.dot(self.a[i], self.weights[i]) + self.biases[i])
            self.a.append(self.sigmoid(self.z[i]))
        return self.a[-1]
    
    def backward(self, X, y):
        m = X.shape[0]
        self.d_weights = []
        self.d_biases = []

        # Compute the error in the output layer
        d_loss = self.a[-1] - y
        d_a = d_loss * self.sigmoid_derivative(self.a[-1])
        
        # Backpropagate the error
        for i in reversed(range(len(self.weights))):
            d_weights = np.dot(self.a[i].T, d_a) / m
            d_biases = np.sum(d_a, axis=0, keepdims=True) / m
            self.d_weights.insert(0, d_weights)
            self.d_biases.insert(0, d_biases)
            if i > 0:
                d_a = np.dot(d_a, self.weights[i].T) * self.sigmoid_derivative(self.a[i])

    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            self.update_parameters()
            if epoch % 100 == 0:
                loss = np.mean((self.a[-1] - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
y = y.reshape(-1, 1)  # Reshape for binary classification

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the network
nn = DeepFeedForwardNN(input_size=2, hidden_layers=[8, 8, 8, 8], output_size=1, learning_rate=0.01)
nn.train(X_train, y_train, epochs=1000)

# Test the network
predictions = nn.predict(X_test)
accuracy = np.mean((predictions > 0.5) == y_test)
print(f'Accuracy: {accuracy:.2f}')
import matplotlib.pyplot as plt

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', marker='o')
    plt.show()

plot_decision_boundary(nn.predict, X_test, y_test)