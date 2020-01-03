from google_drive_downloader import GoogleDriveDownloader as gdd
import os
from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator

# Download datasets from Google Drive
gdd.download_file_from_google_drive(file_id='1zCCpWhDfXVh4dEQoMB09nKSfLNJi-HyT',
                                    dest_path='data/dataset_AMLS_19-20.zip',
                                    unzip=True)

data_dir = "data/dataset_AMLS_19-20" 
os.chdir(data_dir) 
print("Directory has been changed")

celeba_dir = "celeba"
cartoon_set_dir = "cartoon_set"

celeba_img_height = 218
celeba_img_height = 178

def load_gender_training_data(batch_size=32):
    os.chdir(celeba_dir)
    # Import data as dataframe, drop unnecessary column
    df = pd.read_csv("labels.csv", sep="\t", dtype=str)

    # Create separate dataframe for gender
    gender = df.copy()
    gender.drop(gender.columns[0], axis=1, inplace=True)
    
    # Now, we create training and test sets for the gender dataset
    gender_train, gender_test = train_test_split(gender, test_size=0.2, random_state=42)

    # We now create two ImageDataGenerator objects for the gender dataset:
    # one for training, the other for validation
    # See https://forums.fast.ai/t/split-data-using-fit-generator/4380/4
    # for validation split

    # We rescale to ensure RGB values fall between 0 and 1
    # We set aside 20% of the training set for validation
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # We generate an image-label pair for the training set as follows
    gender_train_gen = datagen.flow_from_dataframe(dataframe=gender_train, 
                                                   directory="img/",
                                                   x_col="img_name",
                                                   y_col="gender",
                                                   class_mode="sparse",
                                                   target_size=(218,178),
                                                   batch_size=batch_size,
                                                   subset="training"
    )

    # We generate an image-label pair for the validation set as follows
    gender_val_gen = datagen.flow_from_dataframe(dataframe=gender_train,
                                                 directory="img/",
                                                 x_col="img_name",
                                                 y_col="gender",
                                                 class_mode="sparse",
                                                 target_size=(218,178),
                                                 batch_size=batch_size,
                                                 subset="validation"
    )

    os.chdir(data_dir)

    return gender_train_gen, gender_val_gen

load_gender_training_data()
print("Data has been loaded")
