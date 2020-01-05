# Transfer learning using the Xception architecture pretrained on ImageNet.
# It has been modified to allow a custom input size.

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

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

    # First, we freeze the layers for the first part of the training
    for layer in base_model.layers:
        layer.trainable = False

    # Specify optimiser parameters for the first stage
    optimizer = SGD(lr=0.2, momentum=0.9, decay=0.01)
        
    # We now compile the Xception model for the first stage
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

    # Training and evaluating the Xception model for the first stage
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        epochs=int(epochs/2))

    # Now, we unfreeze the layers for the second part of the training
    for layer in base_model.layers:
        layer.trainable = True

    # Specify optimiser parameters for the second stage
    optimizer = SGD(lr=0.01, momentum=0.9, decay=0.001)
        
    # We now compile the Xception model for the second stage
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

    # Training and evaluating the Xception model for the second stage
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        epochs=int(epochs/2))
    return model, history