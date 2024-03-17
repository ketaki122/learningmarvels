
from google.cloud import storage
from pathlib import Path
import os
import re



# Function to download files from GCS
def download_files(url,category):
    print("Downloading...")
    # Path to your service account JSON file
    path_to_private_key = os.path.abspath("match/fifth-compass-415612-76f634511b19.json") 

    # Initialize the client using the service account JSON file
    client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)

    # Name of the bucket
    bucket_name = ''
    # Extract the bucket name from the URL
    match = re.search(r'\/([^\/]+)\/?$', url)
    if match:
        bucket_name=match.group(1)
        #print(bucket_name)
    else:
        raise ValueError("Invalid URL format. Could not extract bucket name.")
    
    # Get the bucket
    bucket = client.get_bucket(bucket_name)
    # Define the folder names
    #jd_folder_on_gcs = 'JD/'
    #resume_data_folder_on_gcs = 'RESUME/data/'

    # Define local directories to save files
    base_local_dir = './'
    data_local_dir =''
    if category.upper() =='JOB':
        data_local_dir = os.path.join(base_local_dir, 'job-data/')
    elif category.upper() =='RESUME':
        data_local_dir = os.path.join(base_local_dir, 'resume-data/')
    # jd_local_dir = os.path.join(data_local_dir, 'JD/')
    # resume_local_dir = os.path.join(data_local_dir, 'RESUMES/')

    # Make sure local directories exist
    Path(data_local_dir).mkdir(parents=True, exist_ok=True)
    # Path(jd_local_dir).mkdir(parents=True, exist_ok=True)
    # Path(resume_local_dir).mkdir(parents=True, exist_ok=True)
    # Make sure local directories exist
    # Path(jd_local_dir).mkdir(parents=True, exist_ok=True)
    # Path(resume_local_dir).mkdir(parents=True, exist_ok=True)
    blobs = bucket.list_blobs()
    for blob in blobs:
        if not blob.name.endswith('/'):
            # Check if the file has the desired extension
            if (category.upper()=="RESUME"):
                if (blob.name.lower().endswith('.pdf')):
                    # Extract the file name from the blob name
                    file_name = os.path.basename(blob.name)
                    # Download the file to the local directory
                    blob.download_to_filename(os.path.join(data_local_dir, file_name))
                    print(f'Downloaded file [{file_name}] from [{bucket_name}]')
            elif (category.upper()=="JOB"):
                if (blob.name.lower().endswith('.docx')):
                    # Extract the file name from the blob name
                    file_name = os.path.basename(blob.name)
                    # Download the file to the local directory
                    blob.download_to_filename(os.path.join(data_local_dir, file_name))
                    print(f'Downloaded file [{file_name}] from [{bucket_name}]')


    return data_local_dir
    print("Download complete.")


# Download files from gcs
#url="https://console.cloud.google.com/storage/browser/hackathontestdata2024"
#download_files()





