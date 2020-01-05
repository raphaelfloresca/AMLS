import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.datasets.utilities import create_datagen, create_celeba_df

def create_gender_df():
    # Import celeba_df
    celeba_df = create_celeba_df()

    # Create gender dataframe, drop unnecessary columns
    gender_df = celeba_df.copy()
    gender_df.drop(gender_df.columns[0], axis=1, inplace=True)
    gender_df.drop(gender_df.columns[2], axis=1, inplace=True)
    return gender_df

# Create training, validation and test ImageDataGenerator objects
# for the gender data which will be used for training, validation and testing
def create_gender_datagens(height, width, batch_size, test_size, validation_split, random_state):
    # Create datagen
    datagen = create_datagen(validation_split)

    # Get gender dataframe
    gender_df = create_gender_df()    

    # Create training and test sets for the gender and smiling datasets
    gender_train, gender_test = train_test_split(
        gender_df,
        test_size=test_size,
        random_state=random_state)

    # Generate an image-label pair for the training set
    gender_train_gen = datagen.flow_from_dataframe(
        dataframe=gender_train, 
        directory="img/",
        x_col="img_name",
        y_col="gender",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="training")

    # Generate an image-label pair for the validation set as follows
    gender_val_gen = datagen.flow_from_dataframe(
        dataframe=gender_train,
        directory="img/",
        x_col="img_name",
        y_col="gender",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="validation")

    # Generate an image-label pair for the gender test set as follows
    # Set batch_size = size of test set
    gender_test_gen = datagen.flow_from_dataframe(
        dataframe=gender_test,
        directory="img/",
        x_col="img_name",
        y_col="gender",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=len(gender_test))

    return gender_train_gen, gender_val_gen, gender_test_gen

