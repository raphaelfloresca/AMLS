import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.datasets.utilities import create_datagen, create_cartoon_set_df, go_up_three_dirs

def create_face_shape_df():
    # Go up three directories
    go_up_three_dirs()

    # Import cartoon_set_df
    cartoon_set_df = create_cartoon_set_df()

    # Create face_shape dataframe, drop unnecessary columns
    face_shape_df = cartoon_set_df.copy()
    face_shape_df.drop(face_shape_df.columns[0], axis=1, inplace=True)
    face_shape_df.drop(face_shape_df.columns[0], axis=1, inplace=True)
    return face_shape_df

def create_face_shape_datagens(height, width, batch_size, test_size, validation_split, random_state):
    # Create datagen
    datagen = create_datagen(validation_split)

    # Create face shape dataframe
    face_shape_df = create_face_shape_df
    
    # Now, we create training and test sets for the face_shape and smiling datasets
    face_shape_train, face_shape_test = train_test_split(
        face_shape_df,
        test_size=test_size,
        random_state=random_state
    )

    # We generate an image-label pair for the training set as follows
    face_shape_train_gen = datagen.flow_from_dataframe(
        dataframe=face_shape_train, 
        directory="img/",
        x_col="file_name",
        y_col="face_shape",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="training"
    )

    # We generate an image-label pair for the validation set as follows
    face_shape_val_gen = datagen.flow_from_dataframe(
        dataframe=face_shape_train,
        directory="img/",
        x_col="file_name",
        y_col="face_shape",
            class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
        subset="validation"
    )

    # We generate an image-label pair for the face_shape test set as follows
    # We set batch_size = size of test set
    face_shape_test_gen = datagen.flow_from_dataframe(
        dataframe=face_shape_test,
        directory="img/",
        x_col="file_name",
        y_col="face_shape",
        class_mode="sparse",
        target_size=(height,width),
        batch_size=len(face_shape_test)
    )
    return face_shape_train_gen, face_shape_val_gen, face_shape_test_gen