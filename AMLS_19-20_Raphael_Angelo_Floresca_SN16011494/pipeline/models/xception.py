# Transfer learning using the Xception architecture pretrained on ImageNet.
# It has been modified to allow a custom input size.

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from pathlib import Path
from pipeline.optimisation.learning_rate_schedulers import StepDecay, PolynomialDecay
from pipeline.optimisation.one_cycle_lr.lr_finder import LRFinder
from pipeline.optimisation.one_cycle_lr.one_cycle_scheduler import OneCycleScheduler

def train_frozen_xception(
        height,
        width,
        num_classes,
        batch_size,
        epochs,
        learning_rate,
        schedule_type,
        train_gen,
        val_gen,
        frozen_model_path):

    # Store the number of epochs to train for in a convenience variable,
    # then initialize the list of callbacks and learning rate scheduler
    # to be used
    callbacks = []
    schedule = None
    
    # check to see if step-based learning rate decay should be used
    if schedule_type == "step":
        print("[INFO] using 'step-based' learning rate decay...")
        schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=int(epochs/5))

    # check to see if linear learning rate decay should should be used
    elif schedule_type == "linear":
        print("[INFO] using 'linear' learning rate decay...")
        schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)
    
    # check to see if a polynomial learning rate decay should be used
    elif schedule_type == "poly":
        print("[INFO] using 'polynomial' learning rate decay...")
        schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=5)

    elif schedule_type == "one_cycle":
        print("[INFO] using 'one cycle' learning...")
        schedule = OneCycleScheduler(learning_rate)
        callbacks = [schedule]

    # if the learning rate schedule is not empty, add it to the list of
    # callbacks
    if schedule_type != "none" and schedule_type != "one_cycle":
        callbacks = [LearningRateScheduler(schedule)]

    # initialize the decay for the optimizer
    decay = 0.0
 
    # if we are using Keras' "standard" decay, then we need to set the
    # decay parameter
    if schedule_type == "standard":
        print("[INFO] using 'keras standard' learning rate decay...")
        decay = 1e-1 / epochs
 
    # otherwise, no learning rate schedule is being used
    elif schedule_type == "none":
        print("[INFO] no learning rate schedule being used")
        
    # Xception is used as the base architecture for the model.
    # The top layers are not included in order to perform transfer learning.
    # Modified to allow for a custom input size
    base_model = Xception(weights="imagenet",
                          include_top=False,
                          input_shape=(height,width,3))
        
    # Implement own pooling layer
    avg = GlobalAveragePooling2D()(base_model.output)
    
    # Output layer
    output = Dense(num_classes, activation="softmax")(avg)

    # Build model
    frozen_model = Model(inputs=base_model.input, outputs=output)

    # First, we freeze the layers for the first part of the training
    for layer in base_model.layers:
        layer.trainable = False

    if schedule_type != "one_cycle":
        # initialize optimizer and model, then compile it
        opt = SGD(lr=learning_rate, momentum=0.9, decay=decay)
    else:
        opt = SGD()
        
    # We now compile the Xception model for the first stage
    frozen_model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                         metrics=["accuracy"])

    # Training and evaluating the Xception model for the first stage
    frozen_model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        callbacks=callbacks,
        epochs=int(epochs/2))

    # Save model
    frozen_model.save(frozen_model_path)

def train_xception(
        height,
        width,
        num_classes,
        batch_size,
        epochs,
        learning_rate,
        schedule_type,
        find_lr,
        train_gen,
        val_gen,
        frozen_model_path):
    if find_lr == True:
        print("[INFO] Finding learning rate...")
        
        train_frozen_xception(
            height,
            width,
            num_classes,
            batch_size,
            epochs,
            learning_rate,
            schedule_type,
            train_gen,
            val_gen,
            frozen_model_path)

        model = load_model('frozen_model_path')

        lr_finder = LRFinder(model)
        lr_finder.find(train_gen)
        return lr_finder
    else:
        frozen_path = Path("frozen_model_path")

        if frozen_path.is_file() != True:
            train_frozen_xception(
                height,
                width,
                num_classes,
                batch_size,
                epochs,
                learning_rate,
                schedule_type,
                train_gen,
                val_gen,
                frozen_model_path)

        model = load_model('frozen_model_path')

        for layer in model.layers[:-2]:
            layer.trainable = True

        # Store the number of epochs to train for in a convenience variable,
        # then initialize the list of callbacks and learning rate scheduler
        # to be used
        callbacks = []
        schedule = None

        # check to see if step-based learning rate decay should be used
        if schedule_type == "step":
            print("[INFO] using 'step-based' learning rate decay...")
            schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=int(epochs/5))

        # check to see if linear learning rate decay should should be used
        elif schedule_type == "linear":
            print("[INFO] using 'linear' learning rate decay...")
            schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)

        # check to see if a polynomial learning rate decay should be used
        elif schedule_type == "poly":
            print("[INFO] using 'polynomial' learning rate decay...")
            schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=5)

        elif schedule_type == "one_cycle":
            print("[INFO] using 'one cycle' learning...")
            schedule = OneCycleScheduler(learning_rate)
            callbacks = [schedule]

        # if the learning rate schedule is not empty, add it to the list of
        # callbacks
        if schedule_type != "none" and schedule_type != "one_cycle":
            callbacks = [LearningRateScheduler(schedule)]

        # initialize the decay for the optimizer
        decay = 0.0

        # if we are using Keras' "standard" decay, then we need to set the
        # decay parameter
        if schedule_type == "standard":
            print("[INFO] using 'keras standard' learning rate decay...")
            decay = 1e-1 / epochs
    
        # otherwise, no learning rate schedule is being used
        elif schedule_type == "none":
            print("[INFO] no learning rate schedule being used")

        if schedule_type != "one_cycle":
            # initialize optimizer and model, then compile it
            opt = SGD(lr=learning_rate, momentum=0.9, decay=decay)
        else:
            opt = SGD()
    
        # We now compile the Xception model for the first stage
        model.compile(loss="sparse_categorical_crossentropy", 
            optimizer=opt,
            metrics=["accuracy"])

        # Training and evaluating the Xception model for the second stage
        history = model.fit(train_gen,
            steps_per_epoch=train_gen.samples // batch_size,
            validation_data=val_gen,
            validation_steps=val_gen.samples // batch_size,
            callbacks=callbacks,
            epochs=int(epochs/2))
        return model, history, schedule