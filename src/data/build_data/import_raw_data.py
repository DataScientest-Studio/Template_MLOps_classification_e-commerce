import requests
import os
import logging
from src.data.build_data.check_structure import check_existing_file, check_existing_folder
from dotenv import load_dotenv

def import_raw_data(raw_data_relative_path, filenames, bucket_folder_url):
    """import filenames from bucket_folder_url in raw_data_relative_path"""
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    # download all the files
    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f"downloading {input_file} as {os.path.basename(output_file)}")
            response = requests.get(object_url)
            if response.status_code == 200:
                # Process the response content as needed
                content = (
                    response.content
                )  # Utilisez response.content pour les fichiers binaires
                with open(output_file, "wb") as file:
                    file.write(content)
            else:
                print(f"Error accessing the object {input_file}:", response.status_code)

    # Téléchargez le dossier 'img_train'
    img_train_folder_url = os.path.join(bucket_folder_url, "image_train/")
    img_train_local_path = os.path.join(raw_data_relative_path, "image_train/")
    if check_existing_folder(img_train_local_path):
        os.makedirs(img_train_local_path)

    try:
        response = requests.get(img_train_folder_url)
        if response.status_code == 200:
            file_list = response.text.splitlines()
            for img_url in file_list:
                img_filename = os.path.basename(img_url)
                output_file = os.path.join(img_train_local_path, img_filename)
                if check_existing_file(output_file):
                    print(f"downloading {img_url} as {img_filename}")
                    img_response = requests.get(img_url)
                    if img_response.status_code == 200:
                        with open(output_file, "wb") as img_file:
                            img_file.write(img_response.content)
                    else:
                        print(f"Error downloading {img_url}:", img_response.status_code)
        else:
            print(
                f"Error accessing the object list {img_train_folder_url}:",
                response.status_code,
            )
    except Exception as e:
        nonlocalprint(f"An error occurred: {str(e)}")

####################
# Here we are redefining the methods to get the data from Hugging Face
# From now on, we will use the (download_file,unzip_file) methods to get the data
# The code up is not used anymore
####################

# Files list to download
FILES = [
    "X_test_update.csv",
    "X_train_update.csv",
    "Y_train_CVw08PX.csv",
    "images_low.zip",
]

# Load environment variables from .env file
load_dotenv()

# Base URL to download data
BASE_URL = "https://huggingface.co/datasets/ndiguesene/ml-datasets-image-rakuten-ecommerce/resolve/main/"

# Directory where to save the raw data : added /app for dockerization needs
RAW_DATA_PATH = os.environ.get("DATA_PATH", "../../../data/raw")

def download_file(url, output_path):
    """Télécharge un fichier depuis une URL et l'enregistre à output_path."""
    if os.path.exists(output_path):
        logger.info(f"Le fichier {output_path} existe déjà, téléchargement ignoré.")
        return

    logger.info(f"Téléchargement de {url} vers {output_path}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 Mo
                if chunk:
                    file.write(chunk)
        logger.info(f"Téléchargement terminé : {output_path}")
    else:
        logger.error(f"Erreur lors du téléchargement ({response.status_code}): {url}")

def unzip_file(zip_path, extract_to):
    """Décompresse un fichier ZIP vers un dossier donné."""
    extract_folder = os.path.splitext(zip_path)[0]  # Ex: ./data/raw/images_low
    if os.path.exists(extract_folder):
        logger.info(f"Le dossier {extract_folder} existe déjà, extraction ignorée.")
        return

    logger.info(f"Décompression de {zip_path} vers {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Décompression terminée.")

def main_s3(
    raw_data_relative_path="./data/raw",
    filenames=["X_test_update.csv", "X_train_update.csv", "Y_train_CVw08PX.csv"],
    bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/classification_e-commerce/",
):
    """Upload data from AWS s3 in ./data/raw : not used"""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info("making raw data set")

def main():
    """Télécharge les fichiers CSV et ZIP, puis décompresse le fichier ZIP."""
    os.makedirs(RAW_DATA_PATH, exist_ok=True)  # Créer le dossier si nécessaire

    for filename in FILES:
        file_url = BASE_URL + filename
        file_path = os.path.join(RAW_DATA_PATH, filename)

        # Télécharger chaque fichier
        download_file(file_url, file_path)

        # Décompresser le ZIP après téléchargement
        if filename.endswith(".zip"):
            unzip_file(file_path, RAW_DATA_PATH)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
