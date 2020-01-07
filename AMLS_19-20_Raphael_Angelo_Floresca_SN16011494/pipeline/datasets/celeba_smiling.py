import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.datasets.utilities import create_celeba_df

def create_smiling_df():
    # Import celeba_df
    celeba_df = create_celeba_df()

    # Create smiiling dataframe, drop unnecessary columns
    smiling_df = celeba_df.copy()
    smiling_df.drop(smiling_df.columns[0], axis=1, inplace=True)
    smiling_df.drop(smiling_df.columns[1], axis=1, inplace=True)
    return smiling_df