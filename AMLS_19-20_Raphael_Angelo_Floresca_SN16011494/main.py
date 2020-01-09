from A1.a1 import A1MLP, A1CNN
from A2.a2 import A2MLP, A2CNN
from B1.b1 import B1MLP, B1CNN
from B2.b2 import B2MLP, B2CNN
from tensorflow.keras import backend as K
import argparse

# ======================================================================================================================
# Data preprocessing:
# This is done in the respective task packages, using functions from the pipeline.datasets package.
# ======================================================================================================================
# Argument parser, used to define which learning rate scheduler to use as well as the number of epochs
# This section is taken from: https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--schedule-type", type=list, default=["","","",""],
	help="learning rate schedule method")
ap.add_argument("-e", "--epochs", type=int, default=[10,10,10,10],
	help="# of epochs to train for")
ap.add_argument("-l", "--learning-rates", type=list, default=[0.01,0.01,0.01,0.01],
    help="starting learning rate")
ap.add_argument("-f", "--find-lr", type=bool, default=False,
    help="find learning rate with the learning rate finder")
ap.add_argument("-r", "--random-state", type=int, default=None,
    help="random state for splitting the training and test sets, used for replicating results")
ap.add_argument("-t", "--model-type", type=list, default=["cnn","cnn","cnn","cnn"],
    help="choose model type ('mlp', 'cnn', 'xception') for each of the tasks")
args = vars(ap.parse_args())

# ======================================================================================================================
# Task A1
if args.get("model-type")[0] == "mlp":
    model_A1 = A1MLP(args["epochs"][0], args["learning-rates"][0], args["schedule-type"][0], args["random_state"], args["find_lr"])        # Build model object.
elif args.get("model-type")[0] == "cnn":
    model_A1 = A1CNN(args["epochs"][0], args["learning-rates"][0], args["schedule-type"][0], args["random_state"], args["find_lr"])        # Build model object.
acc_A1_train = model_A1.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task A2
if args.get("model-type")[1] == "mlp":
    model_A2 = A2MLP(args["epochs"][1], args["learning-rates"][1], args["schedule-type"][1], args["random_state"], args["find_lr"])        # Build model object.
elif args.get("model-type")[1] == "cnn":
    model_A2 = A2CNN(args["epochs"][1], args["learning-rates"][1], args["schedule-type"][1], args["random_state"], args["find_lr"])        # Build model object.
acc_A2_train = model_A2.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A2_test = model_A2.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task B1
if args.get("model-type")[2] == "mlp":
    model_B1 = B1MLP(args["epochs"][2], args["learning-rates"][2], args["schedule-type"][2], args["random_state"], args["find_lr"])        # Build model object.
elif args.get("model-type")[2] == "cnn":
    model_B1 = B1CNN(args["epochs"][2], args["learning-rates"][2], args["schedule-type"][2], args["random_state"], args["find_lr"])        # Build model object.
acc_B1_train = model_B1.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_B1_test = model_B1.test()   # Test model based on the test set.

# Clear GPU memory
K.clear_session()

# ======================================================================================================================
# Task B2
if args.get("model-type")[3] == "mlp":
    model_B2 = B2MLP(args["epochs"][3], args["learning-rates"][3], args["schedule-type"][3], args["random_state"], args["find_lr"])        # Build model object.
elif args.get("model-type")[3] == "cnn":
    model_B2 = B2CNN(args["epochs"][3], args["learning-rates"][3], args["schedule-type"][3], args["random_state"], args["find_lr"])        # Build model object.
acc_B2_train = model_B2.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_B2_test = model_B2.test()   # Test model based on the test set.

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