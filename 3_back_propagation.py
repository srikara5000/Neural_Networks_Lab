# Build a deep feed forward ANN by implementing the backpropagation algorithm an test the same using appropriate dataset. Use the hidden layers >=4

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DeepFeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, dropout_rate=0.5):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.z.append(z)
            if i == len(self.weights) - 1:  # Output layer
                self.a.append(self.sigmoid(z))
            else:  # Hidden layers
                a = self.relu(z)
                # Apply dropout
                a = a * (np.random.rand(*a.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                self.a.append(a)
        return self.a[-1]

    def backward(self, X, y):
        m = X.shape[0]
        self.d_weights = []
        self.d_biases = []

        # Output layer error
        d_loss = self.a[-1] - y
        d_a = d_loss / m

        for i in reversed(range(len(self.weights))):
            d_weights = np.dot(self.a[i].T, d_a)
            d_biases = np.sum(d_a, axis=0, keepdims=True)
            self.d_weights.insert(0, d_weights)
            self.d_biases.insert(0, d_biases)
            if i > 0:
                d_a = np.dot(d_a, self.weights[i].T) * self.relu_derivative(self.a[i])

    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def compute_loss(self, y_pred, y):
        return -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

    def train(self, X, y, epochs=2000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y)
            self.update_parameters()
            if epoch % 100 == 0:
                loss = self.compute_loss(y_pred, y)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Generate synthetic dataset
X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the network
nn = DeepFeedForwardNN(input_size=2, hidden_layers=[16, 16, 16, 16], output_size=1, learning_rate=0.01, dropout_rate=0.2)
nn.train(X_train, y_train, epochs=2000)

# Test the network
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy:.2f}')

# Plot decision boundary
def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', marker='o')
    plt.show()

plot_decision_boundary(lambda x: nn.predict(x), X_test, y_test)

# Output:
# Epoch 0, Loss: 1.0240
# Epoch 100, Loss: 0.6021
# Epoch 200, Loss: 0.5197
# Epoch 300, Loss: 0.4897
# Epoch 400, Loss: 0.4606
# Epoch 500, Loss: 0.4486
# Epoch 600, Loss: 0.4143
# Epoch 700, Loss: 0.4132
# Epoch 800, Loss: 0.3922
# Epoch 900, Loss: 0.4026
# Epoch 1000, Loss: 0.3850
# Epoch 1100, Loss: 0.3824
# Epoch 1200, Loss: 0.3892
# Epoch 1300, Loss: 0.3714
# Epoch 1400, Loss: 0.3579
# Epoch 1500, Loss: 0.3639
# Epoch 1600, Loss: 0.3775
# Epoch 1700, Loss: 0.3691
# Epoch 1800, Loss: 0.3464
# Epoch 1900, Loss: 0.3470
# Accuracy: 0.84
