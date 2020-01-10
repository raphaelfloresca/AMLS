from google_drive_downloader import GoogleDriveDownloader as gdd

# Download datasets from Google Drive
train_file_id = '1zCCpWhDfXVh4dEQoMB09nKSfLNJi-HyT' 
train_dest_path = 'data/dataset_AMLS_19-20.zip' 

test_file_id = '1SxoyIvITKBxcjsyb0mQwW3qU1_uLK415'
test_dest_path = 'data/dataset_test_AMLS_19-20.zip' 

# Download datasets from Google Drive
def download_train_dataset():
    gdd.download_file_from_google_drive(
        file_id=train_file_id,
        dest_path=train_dest_path,
        unzip=True)

def download_test_dataset():
    gdd.download_file_from_google_drive(
        file_id=test_file_id,
        dest_path=test_dest_path,
        unzip=True)