from A1.a1_cnn import A1_CNN
from A2.a2_cnn import A2_CNN
from B1.b1_cnn import B1_CNN
from B2.b2_cnn import B2_CNN
from tensorflow.keras import backend as K
from numba import cuda



# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
model_A1 = A1_CNN()                     # Build model object.
acc_A1_train = model_A1.train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test()   # Test model based on the test set.

# Clear gpu memory
K.clear_session()
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)

# ======================================================================================================================
# Task A2
model_A2 = A2_CNN()
acc_A2_train = model_A2.train()
acc_A2_test = model_A2.test()

# Clear gpu memory
K.clear_session()
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)

# ======================================================================================================================
# Task B1
model_B1 = B1_CNN()
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test()

# Clear gpu memory
K.clear_session()
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)

# ======================================================================================================================
# Task B2
model_B2 = B2_CNN()
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test()

# Clear gpu memory
K.clear_session()
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'