import boto3
import logging
from botocore.exceptions import ClientError
import os
import json

def aws_sts_login(ARN, SessionName):
    
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
    s3_client = sts_log_session.client("s3")
    return s3_client

def upload_file(s3_client, file_path, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_path: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_path is used
    :return: True if file was uploaded, else False
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
    cfg_file = open(os.path.abspath(".aws_config"),"r")
    cfg = json.load(cfg_file)
    return cfg

def download_file(s3_client, object_name, bucket, file_path = None):
    """Download a file from a S3 bucket to file_path

    :param file_path: File to download
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_path is used
    :return: True if file was uploaded, else False
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

# How to use:
#aws_config = load_aws_cfg("/home/jc/Workspace/mar24cmlops_rakuten/.aws_config")
#sts_session = aws_sts_login(**aws_config)
#s3_client = s3_login(sts_session)
#upload_file(s3_client, file_path, bucket, object_name=None)
#download_file(s3_client,'Y_train.csv', 'rakutenprojectbucket')