from A1.a1_xception import A1_Xception
from A2.a2_xception import A2_Xception
from B1.b1_xception import B1_Xception
from B2.b2_xception import B2_Xception
from tensorflow.keras import backend as K

# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
model_A1 = A1_Xception()                     # Build model object.
acc_A1_train = model_A1.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test()   # Test model based on the test set.

# Clear gpu memory
K.clear_session()

# ======================================================================================================================
# Task A2
model_A2 = A2_Xception()
acc_A2_train = model_A2.train()
acc_A2_test = model_A2.test()

# Clear gpu memory
K.clear_session()

# ======================================================================================================================
# Task B1
model_B1 = B1_Xception()
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test()

# Clear gpu memory
K.clear_session()

# ======================================================================================================================
# Task B2
model_B2 = B2_Xception()
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test()

# Clear gpu memory
K.clear_session()

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'