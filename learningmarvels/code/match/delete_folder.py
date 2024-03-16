import os
import shutil

def delete_folder(category):
    directory_path=''
    if category.upper()=='JOB':
        directory_path=os.path.abspath("job-data")
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            print(f"The directory {directory_path} has been deleted.")
        else:
            print(f"The directory {directory_path} does not exist.")
    elif category.upper()=='RESUME':
        directory_path=os.path.abspath("resume-data")
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            print(f"The directory {directory_path} has been deleted.")
        else:
            print(f"The directory {directory_path} does not exist.")
                                                
#delete_folder("job")