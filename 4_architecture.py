import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# 1. Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalize to [0, 1]
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 3. Build a CNN model with L1 and L2 regularization and Dropout
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))  # L1 & L2 Regularization
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))  # L1 & L2 Regularization
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))  # L1 & L2 Regularization
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting

    model.add(Dense(10, activation='softmax'))  # 10 classes

    return model

# 4. Create the model
model = build_model()

# 5. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Setup ModelCheckpoint to save the model at the end of each epoch
checkpoint = ModelCheckpoint(filepath='cnn_fashion_mnist.keras',  # File to save the model
                             monitor='val_loss',  # Monitor validation loss
                             save_best_only=False,  # Save the model at the end of every epoch
                             verbose=1)

# 7. Train the model with pausing capability
# You can pause by stopping training and then resume from saved weights
initial_epoch = 0  # Start from epoch 0. Change this if resuming.

history = model.fit(X_train, y_train, epochs=15, batch_size=64,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint],
                    initial_epoch=initial_epoch)  # initial_epoch allows resuming

# 8. Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# To resume training later, simply reload the model weights and continue training from the last epoch.