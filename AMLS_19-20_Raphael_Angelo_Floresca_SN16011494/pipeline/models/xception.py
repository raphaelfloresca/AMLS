# Transfer learning using the Xception architecture pretrained on ImageNet.
# It has been modified to allow a custom input size.

from tensorflow.keras.applications.resnet_v2 import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

def train_xception(
        height,
        width,
        num_classes,
        epochs,
        batch_size,
        train_gen,
        val_gen):

    # ResNet-50v2 is used as the base architecture for the model.
    # The top layers are not included in order to perform transfer learning.
    # Modified to allow for a custom input size
    base_model = Xception(weights="imagenet",
                            include_top=False,
                            input_shape=(height, width, 3))
        
    # Implement own pooling layer
    avg = GlobalAveragePooling2D()(base_model.output)
    # Output layer
    output = Dense(num_classes, activation="softmax")(avg)
    # Build model
    model = Model(inputs=base_model.input, outputs=output)
        
    # We now compile the MLP model to specify the loss function
    # and the optimizer to use (SGD)
    model.compile(
        loss="sparse_categorical_crossentropy", # b/c of exclusive, sparse outputs
        optimizer='sgd', # We use SGD to optimise the MLP
        metrics=["accuracy"]) # Used for classifiers

    # Training and evaluating the MLP model on the gender dataset
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        epochs=epochs)
    return model, history