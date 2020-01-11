import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pipeline.datasets.download_data import download_train_dataset, download_test_dataset

dataset_dir = ""
data_dir = "data/dataset_AMLS_19-20"
test_dir = "data/dataset_test_AMLS_19-20"
parent_dir = "AMLSassignment19_20/AMLS_19-20_Raphael_Angelo_Floresca_SN16011494"

# This checks whether the training dataset has been downloaded
def check_train_path():
    if not os.path.exists(data_dir):
        download_train_dataset()

# This checks whether the test dataset has been downloaded
def check_test_path():
    if not os.path.exists(test_dir):
        download_test_dataset()

# Paths of the two datasets
celeba_dir = "celeba"
cartoon_set_dir = "cartoon_set"
celeba_test_dir = "celeba_test"
cartoon_set_test_dir = "cartoon_set_test"

# Used to go up three directories
def go_up_three_dirs():
    os.chdir("..")
    os.chdir("..")
    os.chdir("..")

# Create a dataframe for the celeba labels.csv
def create_celeba_df():
    check_train_path()
    os.chdir(os.path.join(data_dir, celeba_dir))
    # Import data as dataframe
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)
    go_up_three_dirs()
    return df

# Create a dataframe for the cartoon_set labels.csv
def create_cartoon_set_df():
    check_train_path()
    os.chdir(os.path.join(data_dir, cartoon_set_dir))
    # Import data as dataframe
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)
    go_up_three_dirs()
    return df

# Create a test dataframe for the celeba labels.csv
def create_celeba_test_df():
    check_test_path()
    os.chdir(os.path.join(test_dir, celeba_test_dir))
    # Import data as dataframe
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)
    go_up_three_dirs()
    return df

# Create a test dataframe for the cartoon_set labels.csv
def create_cartoon_set_test_df():
    check_test_path()
    os.chdir(os.path.join(test_dir, cartoon_set_test_dir))
    # Import data as dataframe
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)
    go_up_three_dirs()
    return df

# Create ImageDataGenerators for training, validation and testing
# Rescale to ensure RGB values fall between 0 and 1, speeding up training.
# Set aside 20% of the training set for validation by default, this can be changed.
def create_train_datagens(
        height, 
        width,
        train_df,
        img_dir,
        x_col,
        y_col,
        batch_size, 
        random_state,
        preprocessing_function,
        validation_split=0.25):

    # Create datagen
    datagen = ImageDataGenerator(
        rescale=1./255, 
        width_shift_range=[-0.15,0.15],
        height_shift_range=[-0.15,0.15],
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=[0.15,1.15],
        validation_split=validation_split)

    # Create dataframes
    train = train_df

    # Generate an image-label pair for the training set
    train_gen = datagen.flow_from_dataframe(
        dataframe=train, 
        directory="img/",
        x_col=x_col,
        y_col=y_col,
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="training",
        preprocessing_function=preprocessing_function)

    # Generate an image-label pair for the validation set as follows
    val_gen = datagen.flow_from_dataframe(
        dataframe=train,
        directory="img/",
        x_col=x_col,
        y_col=y_col,
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="validation",
        preprocessing_function=preprocessing_function)

    go_up_three_dirs()

    return train_gen, val_gen

# Create ImageDataGenerators for training, validation and testing
# Rescale to ensure RGB values fall between 0 and 1, speeding up training.
# Set aside 20% of the training set for validation by default, this can be changed.
def create_test_datagen(
        height, 
        width,
        test_df,
        img_dir,
        x_col,
        y_col,
        batch_size, 
        random_state,
        preprocessing_function):

    # Create datagen
    datagen = ImageDataGenerator(
        rescale=1./255, 
        width_shift_range=[-0.15,0.15],
        height_shift_range=[-0.15,0.15],
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=[0.15,1.15]
    )

    # Create dataframe
    test = test_df

    # Generate an image-label pair for the smiling test set as follows
    # Set batch_size = size of test set
    test_gen = datagen.flow_from_dataframe(
        dataframe=test,
        directory="img/",
        x_col=x_col,
        y_col=y_col,
        class_mode="sparse",
        target_size=(height,width),
        batch_size=len(test),
        preprocessing_function=preprocessing_function)

    go_up_three_dirs()

    return test_gen

def get_X_y_test_sets(test_gen):
    itr = test_gen
    X_test, y_test = itr.next()
    return X_test, y_test