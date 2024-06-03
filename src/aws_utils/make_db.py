import duckdb
import pandas as pd
from s3_utils import aws_sts_login, load_aws_cfg, s3_login, download_file, upload_file
from datetime import datetime
import numpy as np

def process_listing(listing_csv_path):
    listing_df = pd.read_csv(listing_csv_path, index_col= 0)
    listing_df['listing_id'] = listing_df.index
    listing_df = listing_df.rename(columns={'productid': 'user_prdtypecode'})
    listing_df['model_prdtypecode'] = np.nan
    listing_df['waiting_datetime'] = datetime.now()
    listing_df['validate_datetime'] = datetime.now()
    listing_df['status'] = 'validate'
    listing_df['user'] = 'init_user'
    return(listing_df)

def process_user_table():
    user_data = {
    'username': ['jc','fred','wilfried','init_user'],
    'first_name': ['jc','fred','wilfried','init_user'],
    'hashed_password': ['jc','fred','wilfried','init_user'],
    'access_rights': ['administrator','administrator','administrator','user']
    }
    user_df = pd.DataFrame(user_data)
    return(user_df)

def load_csv_into_duckdb(csv_path, table_name):
    """
    Loads a CSV file into a DuckDB database.

    Args:
    - csv_path (str): The path to the CSV file to be loaded.
    - table_name (str): The name of the table in DuckDB.

    Returns:
    - duckdb_conn: The DuckDB connection.
    """
    df = pd.read_csv(csv_path)
    duckdb_conn = duckdb.connect()
    duckdb_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    return duckdb_conn

def save_duckdb_to_file(duckdb_conn, db_file_path):
    """
    Saves the DuckDB database to a file.

    Args:
    - duckdb_conn: The DuckDB connection.
    - db_file_path (str): The path where the database will be saved.
    """
    duckdb_conn.execute(f"EXPORT DATABASE '{db_file_path}' (FORMAT 'PARQUET')")
    return duckdb_conn

#Usage
#aws_config = load_aws_cfg("/home/jc/Workspace/mar24cmlops_rakuten/.aws_config")


#sts_session = aws_sts_login(**aws_config)


#s3_client = s3_login(sts_session)

#upload_file(s3_client, file_path, bucket, object_name=None)

#download_file(s3_client, 'X_test.csv', 'rakutenprojectbucket', file_path = '/home/jc/Workspace/mar24cmlops_rakuten/X_test.csv')
#duckdb_conn = load_csv_into_duckdb('/home/jc/Workspace/mar24cmlops_rakuten/X_test.csv', 'listings')
#save_duckdb_to_file(duckdb_conn, db_file_path = '/home/jc/Workspace/mar24cmlops_rakuten/db.duckdb')
#duckdb_conn.close()


#con = duckdb.connect()

# Requête SQL
#query = f"SELECT * FROM '{"/home/jc/Workspace/mar24cmlops_rakuten/db.duckdb/listings.parquet"}' LIMIT 10"

# Exécution de la requête
#result = duckdb_conn.sql(query).fetchall()

# Fermer la connexion
#con.close()

listing_df = process_listing('X_test.csv')
user_df = process_user_table()


# # Chemin vers le fichier DuckDB
# db_file_path = '/home/jc/Workspace/mar24cmlops_rakuten/db.duckdb'

# # Connexion à la base de données DuckDB
# con = duckdb.connect(database=db_file_path, read_only=False)

# # Requête SQL
# query = f"CREATE TABLE listings2 AS SELECT * FROM df"

# # Exécution de la requête
# result = con.execute(query).fetchall()