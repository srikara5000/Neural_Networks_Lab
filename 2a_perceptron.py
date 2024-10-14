# 2a. Design a single unit perceptron for classification of a linearly separable binary dataset without using pre-defined models. Use the Perceptron () from sklearn. 
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a synthetic dataset with valid parameters
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, flip_y=0, random_state=42)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Step 3: Initialize the Perceptron model
perceptron = Perceptron(max_iter=1000, eta0=1, random_state=42)

# Step 4: Train the Perceptron model
perceptron.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = perceptron.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Optional: Print the model coefficients and intercept
print(f'Coefficients: {perceptron.coef_}')
print(f'Intercept: {perceptron.intercept_}')
