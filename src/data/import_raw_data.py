import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder

def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''import filenames from bucket_folder_url in raw_data_relative_path'''
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    # download all the files
    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f'downloading {input_file} as {os.path.basename(output_file)}')
            response = requests.get(object_url)
            if response.status_code == 200:
                # Process the response content as needed
                content = response.content  # Utilisez response.content pour les fichiers binaires
                with open(output_file, "wb") as file:
                    file.write(content)
            else:
                print(f'Error accessing the object {input_file}:', response.status_code)

    # Téléchargez le dossier 'img_train'
        img_train_folder_url = os.path.join(bucket_folder_url, 'img_train/')
        img_train_local_path = os.path.join(raw_data_relative_path, 'img_train/')
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
                        print(f'downloading {img_url} as {img_filename}')
                        img_response = requests.get(img_url)
                        if img_response.status_code == 200:
                            with open(output_file, "wb") as img_file:
                                img_file.write(img_response.content)
                        else:
                            print(f'Error downloading {img_url}:', img_response.status_code)
            else:
                print(f'Error accessing the object list {img_train_folder_url}:', response.status_code)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main(raw_data_relative_path="../../data/raw", 
        filenames=["X_test_update.csv", "X_train_update.csv", "Y_train_CVw08PX.csv"],
        bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/classification_e-commerce/"
        ):
    """ Upload data from AWS s3 in ./data/raw
    """
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
