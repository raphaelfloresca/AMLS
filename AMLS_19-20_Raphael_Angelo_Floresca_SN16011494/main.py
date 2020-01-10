from A1.a1 import A1MLP, A1CNN, A1Xception
from A2.a2 import A2MLP, A2CNN, A2Xception
from B1.b1 import B1MLP, B1CNN, B1Xception
from B2.b2 import B2MLP, B2CNN, B2Xception
from tensorflow.keras import backend as K
import argparse

# ======================================================================================================================
# Data preprocessing:
# This is done in the respective task packages, using functions from the pipeline.datasets package.
# ======================================================================================================================
# Argument parser, used to define which learning rate scheduler to use as well as the number of epochs
# This section is taken from: https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--schedule_type", type=str, default='none,none,none,none',
	help="learning rate schedule method")
ap.add_argument("-e", "--epochs", type=str, default='10,10,10,10',
	help="# of epochs to train for")
ap.add_argument("-l", "--learning_rates", type=str, default='0.003,0.003,0.003,0.003',
    help="starting learning rate")
ap.add_argument("-f", "--find_lr", type=bool, default=False,
    help="find learning rate with the learning rate finder")
ap.add_argument("-r", "--random_state", type=int, default=None,
    help="random state for splitting the training and test sets, used for replicating results")
ap.add_argument("-t", "--model_type", type=str, default='xception,xception,xception,xception',
    help="choose model type ('mlp', 'cnn', 'xception') for each of the tasks")
args = vars(ap.parse_args())
schedule_type = [str(item) for item in args["schedule_type"].split(",")]
epochs = [int(item) for item in args["epochs"].split(",")]
learning_rates = [float(item) for item in args["learning_rates"].split(",")]
model_type = [str(item) for item in args["model_type"].split(",")]

# ======================================================================================================================
# Task A1
if model_type[0] == "mlp":
    model_A1 = A1MLP(epochs[0], learning_rates[0], schedule_type[0], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[0] == "cnn":
    model_A1 = A1CNN(epochs[0], learning_rates[0], schedule_type[0], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[0] == "xception":
    model_A1 = A1Xception(epochs[0], learning_rates[0], schedule_type[0], args["find_lr"], args["random_state"])   # Build model object.
acc_A1_train = model_A1.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
if args["find_lr"] != True:
    acc_A1_test = model_A1.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task A2
if model_type[1] == "mlp":
    model_A2 = A2MLP(epochs[1], learning_rates[1], schedule_type[1], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[1] == "cnn":
    model_A2 = A2CNN(epochs[1], learning_rates[1], schedule_type[1], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[1] == "xception":
    model_A2 = A2Xception(epochs[1], learning_rates[1], schedule_type[1], args["find_lr"], args["random_state"])   # Build model object.
acc_A2_train = model_A2.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
if args["find_lr"] != True:
    acc_A2_test = model_A2.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task B1
if model_type[2] == "mlp":
    model_B1 = B1MLP(epochs[2], learning_rates[2], schedule_type[2], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[2] == "cnn":
    model_B1 = B1CNN(epochs[2], learning_rates[2], schedule_type[2], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[2] == "xception":
    model_B1 = B1Xception(epochs[2], learning_rates[2], schedule_type[2], args["find_lr"], args["random_state"])   # Build model object.
acc_B1_train = model_B1.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
if args["find_lr"] != True:
    acc_B1_test = model_B1.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task B2
if model_type[3] == "mlp":
    model_B2 = B2MLP(epochs[3], learning_rates[3], schedule_type[3], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[3] == "cnn":
    model_B2 = B2CNN(epochs[3], learning_rates[3], schedule_type[3], args["find_lr"], args["random_state"])        # Build model object.
elif model_type[3] == "xception":
    model_B2 = B2Xception(epochs[3], learning_rates[3], schedule_type[3], args["find_lr"], args["random_state"])   # Build model object.
acc_B2_train = model_B2.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
if args["find_lr"] != True:
    acc_B2_test = model_B2.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Print out your results with following format:
print('TA1:{},{};TA2:{},{};TA3:{},{};TA4:{},{}'.format(acc_A1_train, acc_A1_test,
                                                       acc_A2_train, acc_A2_test,
                                                       acc_B1_train, acc_B1_test,
                                                       acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'