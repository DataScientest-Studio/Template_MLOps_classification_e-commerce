import boto3
import logging
from botocore.exceptions import ClientError
import os

session = boto3.Session()

sts = session.client("sts")

response = sts.assume_role(
    RoleArn="arn:aws:iam::875047674263:role/RakutenProjectAppAccess",
    RoleSessionName="RakutenProjectAppAccess"
)
new_session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                      aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                      aws_session_token=response['Credentials']['SessionToken'])

s3 = new_session.client("s3")

s3.upload_file("/home/jc/Workspace/mar24cmlops_rakuten/src/1.txt", 'rakutenprojectbucket', '1.txt')


# def aws_sts_login(ARN, SessionName):
    
#     session = boto3.Session()

#     sts_client = session.client("sts")

#     response = sts_client.assume_role(
#                                 RoleArn=ARN,
#                                 RoleSessionName=SessionName
#                                 )
    
#     sts_log_session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
#                       aws_secret_access_key=response['Credentials']['SecretAccessKey'],
#                       aws_session_token=response['Credentials']['SessionToken'])
    
#     return sts_log_session

# def s3_login (sts_log_session):
#     s3_client = sts_session.client("s3")
#     return s3_client

# def upload_file(s3_client, file_name, bucket, object_name=None):
#     """Upload a file to an S3 bucket

#     :param file_name: File to upload
#     :param bucket: Bucket to upload to
#     :param object_name: S3 object name. If not specified then file_name is used
#     :return: True if file was uploaded, else False
#     """

#     # If S3 object_name was not specified, use file_name
#     if object_name is None:
#         object_name = os.path.basename(file_name)

#     # Upload the file
    
#     try:
#         response = s3_client.upload_file(file_name, bucket, object_name)
#     except ClientError as e:
#         logging.error(e)
#         return False
#     return True

# sts_session = aws_sts_login("arn:aws:iam::875047674263:role/RakutenProjectAppAccess","RakutenProjectAppAccess")
# s3_client = s3_login(sts_session)

# upload_file(s3_client,'hello4.txt', 'rakutenprojectbucket')

# #s3_client.upload_file('./hello4.txt', 'rakutenprojectbucket')
