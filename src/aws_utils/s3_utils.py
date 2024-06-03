import boto3
import logging
from botocore.exceptions import ClientError
import os
import json

def aws_sts_login(ARN, SessionName):
    """
    Function to assume an AWS role and create an AWS STS session.
    Args:
        ARN: Amazon Resource Name of the role to assume
        SessionName: Name of the session to assume

    Returns:
        boto3 session with new credentials
    """
    session = boto3.Session()

    sts_client = session.client("sts")

    response = sts_client.assume_role(
                                RoleArn=ARN,
                                RoleSessionName=SessionName
                                )
    
    sts_log_session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                      aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                      aws_session_token=response['Credentials']['SessionToken'])
    
    return sts_log_session

def s3_login (sts_log_session):
    """
    Function to create an S3 client with the logged STS session.
    Args:
        sts_log_session: STS session with temporary credentials :return: S3 client
    Returns:
        boto3 session with new credentials
    """
    s3_client = sts_log_session.client("s3")
    return s3_client

def upload_file(s3_client, file_path, bucket, object_name=None):
    """
    Upload a file to an S3 bucket
    Args:
        file_path: File to upload
        bucket: Bucket to upload to
        object_name: S3 object name. If not specified then file_path is used
    Returns: 
        True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_path
    
    if object_name is None:
        object_name = os.path.basename(file_path)

    # Upload the file
    
    try:
        response = s3_client.upload_file(file_path, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def load_aws_cfg(cfg_path):
    """
    Function to load AWS configuration from a JSON file.
    Args:
        cfg_path: Path to configuration file

    Returns:
        Dictionary containing AWS configurations
    """
    cfg_file = open(os.path.abspath(".aws_config"),"r")
    cfg = json.load(cfg_file)
    return cfg

def download_file(s3_client, object_name, bucket, file_path = None):
    """
    Function to download a file from an S3 bucket.
    Args:
        s3_client: S3 client
        object_name: S3 object name. If not specified, use file_path.
        bucket: S3 bucket name
        file_path: Destination path of the downloaded file

    Returns:
        True if download successful, False otherwise
    """
    # If file_path was not specified, use current directory 
    if file_path is None:
        file_path = os.getcwd() + f'/{object_name}'

    # Download the file
    
    try:
        response = s3_client.download_file(bucket,object_name,file_path)
    except ClientError as e:
        logging.error(e)
        return False
    return True

# Example Use :
# Loading AWS configuration from a file
# aws_config = load_aws_cfg("/home/jc/Workspace/mar24cmlops_rakuten/.aws_config")

# Creating an STS session with credentials
# sts_session = aws_sts_login(**aws_config)

# Creating an S3 client with the STS session
# s3_client = s3_login(sts_session)

# Upload a file to the S3 bucket 
# upload_file(s3_client, file_path, bucket, object_name=None)

# Download a file from the S3 bucket 
# download_file(s3_client, file_path, bucket, object_name=None)