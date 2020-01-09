import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pipeline.datasets.download_data import download_dataset
from sklearn.model_selection import train_test_split


data_dir = "data/dataset_AMLS_19-20"
parent_dir = "AMLSassignment19_20/AMLS_19-20_Raphael_Angelo_Floresca_SN16011494"

# This checks whether the dataset has been downloaded
def check_path():
    if not os.path.exists(data_dir):
        download_dataset()

# Paths of the two datasets
celeba_dir = "celeba"
cartoon_set_dir = "cartoon_set"

# Used to go up three directories
def go_up_three_dirs():
    os.chdir("..")
    os.chdir("..")
    os.chdir("..")

# Create a dataframe for the celeba labels.csv
def create_celeba_df():
    check_path()
    os.chdir(os.path.join(data_dir, celeba_dir))
    # Import data as dataframe
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)
    return df

# Create a dataframe for the cartoon_set labels.csv
def create_cartoon_set_df():
    check_path()
    os.chdir(os.path.join(data_dir, cartoon_set_dir))
    # Import data as dataframe
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)
    return df

# Create ImageDataGenerators for training, validation and testing
# Rescale to ensure RGB values fall between 0 and 1, speeding up training.
# Set aside 20% of the training set for validation by default, this can be changed.
def create_datagens(
        height, 
        width,
        df,
        img_dir,
        x_col,
        y_col,
        batch_size, 
        random_state,
        preprocessing_function,
        test_size=0.2,
        validation_split=0.25):

    # Create datagen
    datagen = ImageDataGenerator(
        rescale=1./255, 
        width_shift_range=[-75,75],
        height_shift_range=[-75,75],
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        brightness_range=[0.2,1.0],
        zoom_range=[0.5,1.5],
        validation_split=validation_split)

    # Create dataframe
    df = df

    # Create training and test sets for the smiling and smiling datasets
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

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

    return train_gen, val_gen, test_gen

def get_X_y_test_sets(test_gen):
    itr = test_gen
    gender_X_test, gender_y_test = itr.next()
    return gender_X_test, gender_y_test