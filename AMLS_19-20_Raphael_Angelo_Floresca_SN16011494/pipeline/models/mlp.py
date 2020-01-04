# Creating a classification MLP with two hidden layers
# We are using the Sequential API which creates a stack of layers
# in which the input flows through one after the other

from keras.models import Sequential
from keras.layers import Dense, Flatten

class MLP:
    @staticmethod
    def build(height, width, num_classes, first_af, second_af, layer1_hn, layer2_hn):
        # Instantiate Sequential API
        model = Sequential([
            # This flattens the 218x178x3 input into a 1D tensor
            Flatten(input_shape=(height,width,3)),
            # Fully connected layer with 300 neurons, using ReLU by default.
            Dense(layer1_hn, activation=first_af),
            # Fully connected layer with 100 neurons, using ReLU by default.
            Dense(layer2_hn, activation=second_af),
            # Output layer
            Dense(num_classes, activation="softmax") 
        ])
        return model