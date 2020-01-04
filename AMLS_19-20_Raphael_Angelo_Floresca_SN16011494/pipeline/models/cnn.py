# We create a simple CNN architecture for image classification
# Architecture inspired by:
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb#scrollTo=L1WtoaOHVrVh

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

class CNN:
    @staticmethod
    def build(height, width, num_classes, num_start_filters=16, kernel_size=3, fcl_size=512):
        # Instantiate Sequential API
        model = Sequential([
            # Use 16 3x3 kernels with ReLU after.
            Conv2D(num_start_filters, kernel_size, padding='same', activation='relu', input_shape=(height,width,3)),
            # Pooling layer
            MaxPooling2D(),
            # Use 32 3x3 kernels with ReLU after. Notice this is double the last layer.
            Conv2D(num_start_filters*2, kernel_size, padding='same', activation='relu'),
            # Pooling layer
            MaxPooling2D(),
            # Use 64 3x3 kernels with ReLU after. Notice this is double the last layer.
            Conv2D(num_start_filters*4, kernel_size, padding='same', activation='relu'),
            # Pooling layer
            MaxPooling2D(),
            # Flatten for use with fully-connected layers
            Flatten(),
            # Fully connected layer with 512 neurons
            Dense(fcl_size, activation='relu'),
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        return model