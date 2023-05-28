import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The data has already been splitted into train and test set
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Normalize train/test data
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

# Set random seed
tf.random.set_seed(42)

# Create a model
model = tf.keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.002),
                 metrics=["accuracy"])

# Fit the model
history = model.fit(train_data_norm,
                        train_labels,
                        epochs=10,
                        validation_data=(test_data_norm, test_labels))