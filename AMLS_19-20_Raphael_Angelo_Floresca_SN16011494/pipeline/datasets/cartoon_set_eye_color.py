import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.datasets.utilities import create_cartoon_set_df, create_cartoon_set_test_df

def create_eye_color_df():
    # Import cartoon_set_df
    cartoon_set_df = create_cartoon_set_df()

    # Create eye_color dataframe, drop unnecessary columns
    eye_color_df = cartoon_set_df.copy()
    eye_color_df.drop(eye_color_df.columns[0], axis=1, inplace=True)
    eye_color_df.drop(eye_color_df.columns[1], axis=1, inplace=True)
    return eye_color_df

def create_eye_color_test_df():
    # Import cartoon_set_df
    cartoon_set_df = create_cartoon_set_test_df()

    # Create eye_color dataframe, drop unnecessary columns
    eye_color_df = cartoon_set_df.copy()
    eye_color_df.drop(eye_color_df.columns[0], axis=1, inplace=True)
    eye_color_df.drop(eye_color_df.columns[1], axis=1, inplace=True)
    return eye_color_df