from A1.a1_cnn import A1_CNN
from A2.a2_cnn import A2_CNN
from B1.b1_cnn import B1_CNN
from B2.b2_cnn import B2_CNN
from tensorflow.keras import backend as K
import argparse

# ======================================================================================================================
# Data preprocessing:
# This is done in the respective task packages, using functions from the pipeline.datasets package.
# ======================================================================================================================
# Argument parser, used to define which learning rate scheduler to use as well as the number of epochs
# This section is taken from: https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--schedule", type=str, default="",
	help="learning rate schedule method")
ap.add_argument("-e", "--epochs", type=int, default=10,
	help="# of epochs to train for")
ap.add_argument("-l", "--learning_rate", type=float, default=0.01,
    help="starting learning rate")
args = vars(ap.parse_args())

# ======================================================================================================================
# Task A1
model_A1 = A1_CNN(args["epochs"], args["learning_rate"], args["schedule"])        # Build model object.
acc_A1_train = model_A1.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task A2
model_A2 = A2_CNN(args["epochs"], args["learning_rate"], args["schedule"])        # Build model object.
acc_A2_train = model_A2.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A2_test = model_A2.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task B1
model_B1 = B1_CNN(args["epochs"], args["learning_rate"], args["schedule"])
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test()

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task B2
model_B2 = B2_CNN(args["epochs"], args["learning_rate"], args["schedule"])
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test()

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TA3:{},{};TA4:{},{}'.format(acc_A1_train, acc_A1_test,
                                                       acc_A2_train, acc_A2_test,
                                                       acc_B1_train, acc_B1_test,
                                                       acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'