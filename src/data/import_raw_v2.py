import os
from check_structure import  check_existing_folder
import kagglehub
import shutil

###Set up Kaggle API credentials:
###Navigate to Account Settings → Scroll to API
###Click "Create New API Token" → This downloads a kaggle.json file
###Move kaggle.json to ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

raw_data_relative_path="./data/raw"
dataset = "parth70142/rakuten"

# Download latest version
path = kagglehub.dataset_download(dataset, force_download=True)
print("Path to dataset files:", path)
#final_path = os.path.join(path, " ")
#print("Path to dataset files:", final_path)

if os.path.exists(raw_data_relative_path) is True:    
    shutil.rmtree(raw_data_relative_path)
    shutil.copytree(path, raw_data_relative_path)
    print("Données importées")
else:
    shutil.copytree(path, raw_data_relative_path)
    print("Données importées")
