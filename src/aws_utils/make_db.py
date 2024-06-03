import duckdb
import pandas as pd
from s3_utils import aws_sts_login, load_aws_cfg, s3_login, download_file, upload_file
from datetime import datetime
import numpy as np
import os

def process_listing(listing_csv_path):
    """
    Process the listing CSV file.

    Args:
    - listing_csv_path (str): Path to the listing CSV file.

    Returns:
    - listing_df (pd.DataFrame): Processed DataFrame containing listings data.
    """
    listing_df = pd.read_csv(listing_csv_path, index_col= 0)
    listing_df['listing_id'] = listing_df.index
    listing_df = listing_df.rename(columns={'productid': 'user_prdtypecode'})
    listing_df['model_prdtypecode'] = np.nan
    listing_df['waiting_datetime'] = datetime.now()
    listing_df['validate_datetime'] = datetime.now()
    listing_df['status'] = 'validate'
    listing_df['user'] = 'init_user'
    return(listing_df)

def init_user_table():
    """
    Initialize the user table with default data.

    Returns:
    - user_df (pd.DataFrame): DataFrame containing user data.
    """
    user_data = {
    'username': ['jc','fred','wilfried','init_user'],
    'first_name': ['jc','fred','wilfried','init_user'],
    'hashed_password': ['jc','fred','wilfried','init_user'],
    'access_rights': ['administrator','administrator','administrator','user']
    }
    user_df = pd.DataFrame(user_data)
    return(user_df)
    
def create_table_from_pd_into_duckdb(duckdb_connection,pd_df, table_name):
    """
    Loads a CSV file into a DuckDB database.

    Args:
    - duckdb_connection (duckdb): The DuckDB connection.
    - pd_df (pd.DataFrame): The pd.DataFrame to be loaded.
    - table_name (str): The name of the table in DuckDB.

    Returns:
        None
    """
    duckdb_connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM pd_df")

def save_duckdb_to_parquet(duckdb_conn, db_file_path):
    """
    Saves the DuckDB database to a parquetfile.

    Args:
    - duckdb_conn: The DuckDB connection.
    - db_file_path (str): The path where the database will be saved.
    
    Returns:
        None
    """
    duckdb_conn.execute(f"EXPORT DATABASE '{db_file_path}' (FORMAT 'PARQUET')")

def upload_db_to_s3(aws_config_path, file_path_db, bucket_name):
    """
    Uploads a database file to the "db" folder in an S3 bucket.

    Args:
        aws_config_path (str): Path to the AWS configuration file.
        file_path_db (str): Path to the database file to upload.
        bucket_name (str): Name of the S3 bucket.

    Returns:
        bool: True if the upload was successful, False otherwise.
    """
    # Load AWS configuration from the specified file path
    aws_config = load_aws_cfg(aws_config_path)
    
    # Create an STS session with credentials
    sts_session = aws_sts_login(**aws_config)
    
    # Create an S3 client with the STS session
    s3_client = s3_login(sts_session)
    
    # Define the object name (S3 key) for the uploaded file
    object_name = f"db/{os.path.basename(file_path_db)}"
    
    # Upload the database file to the S3 bucket
    return upload_file(s3_client, file_path_db, bucket_name, object_name)

def download_db_from_s3(aws_config_path, db_file_name, bucket_name, destination_path):
    """
    Downloads a database file from the 'db' directory in an S3 bucket.

    Args:
        aws_config_path (str): Path to the AWS configuration file.
        db_file_name (str): Name of the database file to download.
        bucket_name (str): Name of the S3 bucket.
        destination_path (str): Where to save the DB File

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    # Load AWS configuration from the specified file path
    aws_config = load_aws_cfg(aws_config_path)
    
    # Create an STS session with credentials
    sts_session = aws_sts_login(**aws_config)
    
    # Create an S3 client with the STS session
    s3_client = s3_login(sts_session)
    
    # Define the object name (S3 key) for the database file
    object_name = f"db/{db_file_name}"
    
    # Download the database file from the S3 bucket
    return download_file(s3_client, object_name, bucket_name, file_path=destination_path)

# Example Usage 

# (Init of Rakuten DB, saved on S3)

# listing_df = process_listing('X_train.csv')
# user_df = init_user_table()
# db_file_path = '/home/jc/Workspace/mar24cmlops_rakuten/data/rakuten_db.duckdb'
# con = duckdb.connect(database=db_file_path, read_only=False)
# create_table_from_pd_into_duckdb(con, listing_df, 'fact_listings')
# create_table_from_pd_into_duckdb(con, user_df, 'dim_user')
# con.close()

# (Download Rakuten DB from S3)
# download_db_from_s3('/home/jc/Workspace/mar24cmlops_rakuten/.aws/.aws_config', 
#                     'rakuten_db.duckdb',
#                     'rakutenprojectbucket',
#                     '/home/jc/Workspace/mar24cmlops_rakuten/rakuten_db.duckdb')

# (Upload Rakuten DB from S3)
# upload_db_to_s3('/home/jc/Workspace/mar24cmlops_rakuten/.aws/.aws_config', 
#                 '/home/jc/Workspace/mar24cmlops_rakuten/data/rakuten_db.duckdb',
#                 'rakutenprojectbucket')