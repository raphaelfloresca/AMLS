import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import create_datagen, create_cartoon_set_df

def create_eye_color_datagens(batch_size=32, test_size=0.2, validation_split=0.2, random_state=42):
    # Import cartoon_set_df
    cartoon_set_df = create_cartoon_set_df()

    # Create datagen
    datagen = create_datagen(validation_split)

    # Create eye_color dataframe, drop unnecessary columns
    eye_color = cartoon_set_df.copy()
    eye_color.drop(eye_color.columns[0], axis=1, inplace=True)
    eye_color.drop(eye_color.columns[1], axis=1, inplace=True)

    # Now, we create training and test sets for the eye_color and smiling datasets
    eye_color_train, eye_color_test = train_test_split(
        eye_color,
        test_size=test_size,
        random_state=random_state
    )

    # We generate an image-label pair for the training set as follows
    eye_color_train_gen = datagen.flow_from_dataframe(
        dataframe=eye_color_train, 
        directory="img/",
        x_col="img_name",
        y_col="eye_color",
        class_mode="sparse",
        target_size=(500,500),
        batch_size=32,
        subset="training"
    )

    # We generate an image-label pair for the validation set as follows
    eye_color_val_gen = datagen.flow_from_dataframe(
        dataframe=eye_color_train,
        directory="img/",
        x_col="img_name",
        y_col="eye_color",
            class_mode="sparse",
        target_size=(500,500),
        batch_size=32,
        subset="validation"
    )

    # We generate an image-label pair for the eye_color test set as follows
    # We set batch_size = size of test set
    eye_color_test_gen = datagen.flow_from_dataframe(
        dataframe=eye_color_test,
        directory="img/",
        x_col="img_name",
        y_col="eye_color",
        class_mode="sparse",
        target_size=(500,500),
        batch_size=len(eye_color_test)
    )
    return eye_color_train_gen, eye_color_val_gen, eye_color_test_gen