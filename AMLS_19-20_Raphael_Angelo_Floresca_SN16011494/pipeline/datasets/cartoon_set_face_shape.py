import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import create_datagen, create_cartoon_set_df

def create_face_shape_datagens(batch_size=32, test_size=0.2, validation_split=0.2, random_state=42):
    # Import cartoon_set_df
    cartoon_set_df = create_cartoon_set_df()

    # Create datagen
    datagen = create_datagen(validation_split)

    # Create face_shape dataframe, drop unnecessary columns
    face_shape = cartoon_set_df.copy()
    face_shape.drop(face_shape.columns[0], axis=1, inplace=True)
    face_shape.drop(face_shape.columns[0], axis=1, inplace=True)

    # Now, we create training and test sets for the face_shape and smiling datasets
    face_shape_train, face_shape_test = train_test_split(
        face_shape,
        test_size=test_size,
        random_state=random_state
    )

    # We generate an image-label pair for the training set as follows
    face_shape_train_gen = datagen.flow_from_dataframe(
        dataframe=face_shape_train, 
        directory="img/",
        x_col="img_name",
        y_col="face_shape",
        class_mode="sparse",
        target_size=(500,500),
        batch_size=32,
        subset="training"
    )

    # We generate an image-label pair for the validation set as follows
    face_shape_val_gen = datagen.flow_from_dataframe(
        dataframe=face_shape_train,
        directory="img/",
        x_col="img_name",
        y_col="face_shape",
            class_mode="sparse",
        target_size=(500,500),
        batch_size=32,
        subset="validation"
    )

    # We generate an image-label pair for the face_shape test set as follows
    # We set batch_size = size of test set
    face_shape_test_gen = datagen.flow_from_dataframe(
        dataframe=face_shape_test,
        directory="img/",
        x_col="img_name",
        y_col="face_shape",
        class_mode="sparse",
        target_size=(500,500),
        batch_size=len(face_shape_test)
    )
    return face_shape_train_gen, face_shape_val_gen, face_shape_test_gen