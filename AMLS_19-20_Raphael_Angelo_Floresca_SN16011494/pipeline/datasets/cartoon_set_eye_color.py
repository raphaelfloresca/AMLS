import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.datasets.utilities import create_cartoon_set_df

def create_eye_color_df():
    # Import cartoon_set_df
    cartoon_set_df = create_cartoon_set_df()

    # Create eye_color dataframe, drop unnecessary columns
    eye_color_df = cartoon_set_df.copy()
    eye_color_df.drop(eye_color_df.columns[0], axis=1, inplace=True)
    eye_color_df.drop(eye_color_df.columns[1], axis=1, inplace=True)
    return eye_color_df

def create_eye_color_datagens(
        height, 
        width, 
        batch_size, 
        test_size, 
        validation_split, 
        random_state,
        preprocessing_function):

    # Create datagen
    datagen = create_datagen(validation_split)

    # Create face shape dataframe
    eye_color_df = create_eye_color_df()
    
    # Now, we create training and test sets for the eye_color and smiling datasets
    eye_color_train, eye_color_test = train_test_split(
        eye_color_df,
        test_size=test_size,
        random_state=random_state
    )

    # We generate an image-label pair for the training set as follows
    eye_color_train_gen = datagen.flow_from_dataframe(
        dataframe=eye_color_train, 
        directory="img/",
        x_col="file_name",
        y_col="eye_color",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="training",
        preprocessing_function=preprocessing_function)

    # We generate an image-label pair for the validation set as follows
    eye_color_val_gen = datagen.flow_from_dataframe(
        dataframe=eye_color_train,
        directory="img/",
        x_col="file_name",
        y_col="eye_color",
            class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="validation",
        preprocessing_function=preprocessing_function)

    # We generate an image-label pair for the eye_color test set as follows
    # We set batch_size = size of test set
    eye_color_test_gen = datagen.flow_from_dataframe(
        dataframe=eye_color_test,
        directory="img/",
        x_col="file_name",
        y_col="eye_color",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=len(eye_color_test),
        preprocessing_function=preprocessing_function)
        
    return eye_color_train_gen, eye_color_val_gen, eye_color_test_gen