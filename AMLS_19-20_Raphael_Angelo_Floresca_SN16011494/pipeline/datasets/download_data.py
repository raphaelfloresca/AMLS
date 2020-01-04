from google_drive_downloader import GoogleDriveDownloader as gdd

# Download datasets from Google Drive
file_id = '1zCCpWhDfXVh4dEQoMB09nKSfLNJi-HyT' 
dest_path = 'data/dataset_AMLS_19-20.zip' 

# Download datasets from Google Drive
def download_dataset():
    gdd.download_file_from_google_drive(
        file_id=file_id,
        dest_path=dest_path,
        unzip=True)