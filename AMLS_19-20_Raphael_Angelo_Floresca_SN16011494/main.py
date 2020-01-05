from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
model_A1 = A1()                                     # Build model object.
acc_A1_train = model_A1.mlp_train()                 # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.mlp_test()                  # Test model based on the test set.

# ======================================================================================================================
# Task A2
model_A2 = A2()
acc_A2_train = model_A2.mlp_train()
acc_A2_test = model_A2.mlp_test()

# ======================================================================================================================
# Task B1
model_B1 = B1()
acc_B1_train = model_B1.mlp_train()
acc_B1_test = model_B1.mlp_test()

# ======================================================================================================================
# Task B2
model_B2 = B2()
acc_B2_train = model_B2.mlp_train()
acc_B2_test = model_B2.mlp_test()

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'