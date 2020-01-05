import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.datasets.utilities import create_datagen, create_celeba_df, go_up_three_dirs

def create_smiling_df():
    # Go up three directories
    go_up_three_dirs()
    
    # Import celeba_df
    celeba_df = create_celeba_df()

    # Create smiiling dataframe, drop unnecessary columns
    smiling_df = celeba_df.copy()
    smiling_df.drop(smiling_df.columns[0], axis=1, inplace=True)
    smiling_df.drop(smiling_df.columns[1], axis=1, inplace=True)
    return smiling_df

# Create training, validation and test ImageDataGenerator objects
# for the smiling data which will be used for training, validation and testing
def create_smiling_datagens(height, width, batch_size, test_size, validation_split, random_state):
    # Create datagen
    datagen = create_datagen(validation_split)

    # Create smiling dataframe
    smiling_df = create_smiling_df() 

    # Create training and test sets for the smiling and smiling datasets
    smiling_train, smiling_test = train_test_split(
        smiling_df,
        test_size=test_size,
        random_state=random_state
    )

    # Generate an image-label pair for the training set
    smiling_train_gen = datagen.flow_from_dataframe(
        dataframe=smiling_train, 
        directory="img/",
        x_col="img_name",
        y_col="smiling",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="training"
    )

    # Generate an image-label pair for the validation set as follows
    smiling_val_gen = datagen.flow_from_dataframe(
        dataframe=smiling_train,
        directory="img/",
        x_col="img_name",
        y_col="smiling",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="validation"
    )

    # Generate an image-label pair for the smiling test set as follows
    # Set batch_size = size of test set
    smiling_test_gen = datagen.flow_from_dataframe(
        dataframe=smiling_test,
        directory="img/",
        x_col="img_name",
        y_col="smiling",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=len(smiling_test)
    )
    return smiling_train_gen, smiling_val_gen, smiling_test_gen