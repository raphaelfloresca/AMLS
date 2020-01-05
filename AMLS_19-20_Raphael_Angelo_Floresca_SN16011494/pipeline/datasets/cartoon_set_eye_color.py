import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.datasets.utilities import create_datagen, create_cartoon_set_df, go_up_three_dirs

def create_eye_color_datagens(height, width, batch_size, test_size, validation_split, random_state):
    # Go up three directories
    go_up_three_dirs()
    
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
        target_size=(height,width),
        batch_size=batch_size,
        subset="training"
    )

    # We generate an image-label pair for the validation set as follows
    eye_color_val_gen = datagen.flow_from_dataframe(
        dataframe=eye_color_train,
        directory="img/",
        x_col="img_name",
        y_col="eye_color",
            class_mode="sparse",
        target_size=(height,width),
        batch_size=batch_size,
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
        target_size=(height,width),
        batch_size=len(eye_color_test)
    )
    return eye_color_train_gen, eye_color_val_gen, eye_color_test_gen