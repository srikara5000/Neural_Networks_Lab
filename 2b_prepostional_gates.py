# 2b. Identify the problem with single unit Perceptron. Classify using Or, And and Xor data and analyze the result

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Define datasets
datasets = {
    'OR': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    'AND': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    'XOR': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]))
}

for name, (X, y) in datasets.items():
    # Initialize the Perceptron
    perceptron = Perceptron(max_iter=1000, eta0=1, random_state=42)
    
    # Train the Perceptron
    perceptron.fit(X, y)
    
    # Make predictions
    y_pred = perceptron.predict(X)
    
    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    print(f'{name} Problem:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Coefficients: {perceptron.coef_}')
    print(f'Intercept: {perceptron.intercept_}')
    print()

# Output:
# OR Problem:
# Accuracy: 1.00
# Coefficients: [[2. 2.]]
# Intercept: [-1.]

# AND Problem:
# Accuracy: 1.00
# Coefficients: [[2. 2.]]
# Intercept: [-3.]

# XOR Problem:
# Accuracy: 0.50
# Coefficients: [[0. 0.]]
# Intercept: [0.]