import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pipeline.datasets.download_data import download_dataset

data_dir = "data/dataset_AMLS_19-20" 

# This checks whether the dataset has been downloaded
def check_path():
    if not os.path.exists(data_dir):
        download_dataset()

# Paths of the two datasets
celeba_dir = "celeba"
cartoon_set_dir = "cartoon_set"

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
    os.chdir(os.path.join(data_dir, celeba_dir))
    # Import data as dataframe
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)
    return df

# Create an ImageDataGenerator which will be fed into the
# models for training, validation and testing
# Rescale to ensure RGB values fall between 0 and 1, speeding up training.
# Set aside 20% of the training set for validation by default, this can be changed.
def create_datagen(validation_split=0.2):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    return datagen