# Creating a classification MLP with two hidden layers
# We are using the Sequential API which creates a stack of layers
# in which the input flows through one after the other

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

def train_mlp(
        height, 
        width, 
        num_classes, 
        epochs,
        batch_size, 
        train_gen, 
        val_gen, 
        first_af, 
        second_af, 
        layer1_hn, 
        layer2_hn):
    
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

    # We now compile the MLP model to specify the loss function
    # and the optimizer to use (SGD)
    model.compile(
        loss="sparse_categorical_crossentropy", # b/c of exclusive, sparse outputs
        optimizer='sgd', # We use SGD to optimise the ANN
        metrics=["accuracy"]) # Used for classifiers

    # Training and evaluating the MLP model on the gender dataset
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        epochs=epochs)
    return model, history