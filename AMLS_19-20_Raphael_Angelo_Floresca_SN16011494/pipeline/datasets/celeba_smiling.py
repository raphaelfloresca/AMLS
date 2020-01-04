import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import create_datagen, create_celeba_df

# Create training, validation and test ImageDataGenerator objects
# for the smiling data which will be used for training, validation and testing
def create_smiling_datagens(batch_size=32, test_size=0.2, validation_split=0.2, random_state=42):
    # Import celeba_df
    celeba_df = create_celeba_df()

    # Create datagen
    datagen = create_datagen(validation_split)

    # Create smiling dataframe, drop unnecessary columns
    smiling = celeba_df.copy()
    smiling.drop(smiling.columns[0], axis=1, inplace=True)
    smiling.drop(smiling.columns[2], axis=1, inplace=True)

    # Create training and test sets for the smiling and smiling datasets
    smiling_train, smiling_test = train_test_split(
        smiling,
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
        target_size=(218,178),
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
        target_size=(218,178),
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
        target_size=(218,178),
        batch_size=len(smiling_test)
    )
    return smiling_train_gen, smiling_val_gen, smiling_test_gen