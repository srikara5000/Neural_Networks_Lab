# Design and implement an image classification model to classify a dataset of images using deep feed forward neural network. Record the accuracy corresponding to the number of epochs. Use MNIST datatset.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the Dataset
# Load the MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 2: Design the Model (Deep Feedforward Neural Network)
model = models.Sequential()

# Input layer (784 units corresponding to 28x28 pixels)
model.add(layers.InputLayer(input_shape=(28*28,)))

# Hidden layers (Dense/fully connected layers with ReLU activation)
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))

# Output layer (10 units for 10 classes, with softmax for multi-class classification)
model.add(layers.Dense(10, activation='softmax'))

# Step 3: Compile the Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test), 
                    epochs=5, batch_size=128)

# Step 5: Select your own 3 images from the MNIST test set
# Let's pick 3 random test images
indices = np.random.choice(x_test.shape[0], 3)
sample_images = x_test[indices]
sample_labels = y_test[indices]

# Step 6: Predict the class of the selected images
predictions = model.predict(sample_images)

# Convert predictions from one-hot encoded format to digits
predicted_digits = np.argmax(predictions, axis=1)
actual_digits = np.argmax(sample_labels, axis=1)

# Step 7: Display the selected images and their predicted labels
for i in range(3):
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digits[i]}, Actual: {actual_digits[i]}")
    plt.show()
