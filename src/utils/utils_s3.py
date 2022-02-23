"""
This module contains the utility class for connection to a S3 repository.
Adapted from https://github.com/cshikai/onboarding_demo/blob/main/model/src/aiplatform/s3utility.py
"""
#!/usr/bin/env python
# coding: utf-8

# Standard imports
import os
import sys
import zipfile
import time
from io import BytesIO
from typing import Union
from pathlib import Path

# 3rd party imports
import boto3
import requests
from botocore.client import Config


class S3Utils:
    """
    S3 repo connection utility class

    Attributes
    ----------
    MAX_ATTEMPT: int
        No. attempts to connect to S3
    WAIT_TIME: int
        Seconds to wait per connection attempt
    cert_output_dir: str
        cert output directory
    cert_download_url: str
        url link to download certificate
    bucket: str
        Main S3 Bucket to link to
    endpoint_url : str
        Endpoint URL to private S3 storage
    aws_access_key_id : str
        S3 Access Key (Ask your SA for the key)
    aws_secret_access_key :  str
        S3 Secret Access Key (Ask your SA for the secret)
    signature_version : str
        Version of signature used for S3 Storage (default = 's3v4')
    region_name : str
        Region where the S3 storage is situated (default = 'us-east-1')
    verify : bool | str
        Verify with SSL
        if boolean: False - Do not verify
        if boolean: True - checks the environment variable REQUESTS_CA_BUNDLE for SSL Cert path
        Otherwise, insert SSL Cert path manually as string

    Functions
    ----------
    download_file(bucket_name: str, folder: str, filename: str, local_folder: str)
        Downloads a file from the S3 Storage to a local path.
    download_zipped_file_v1(s3_key: str, local_path: str, bucket: str, temp_dir: str, remove_temp: bool)
        V1 Method - Extracts the contents of a S3 zipped file.
    download_zipped_file_v2(s3_key: str, local_path: str, bucket: str)
        V2 Method - Reads and extracts the contents of a S3 zipped file as a bytes stream.
        (Only works for smaller zip files (<= 2GB), otherwise will overflow)
    download_folder(s3_prefix: str, local_dir: str, bucket: str, verbose: int)
        Downloads all files from S3 Storage with the prefix to a local directory.
    upload_file(local_path: str, s3_key: str, bucket: str, verbose: int)
        Upload a single file to S3
    upload_folder(local_dir: str, s3_prefix: str, bucket: str, verbose: str)
        Uploads all files from a local directory to S3 Storage with the inserted prefix.
    """

    MAX_ATTEMPT = 5
    WAIT_TIME = 1

    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        signature_version: str = "s3v4",
        region_name: str = "us-east-1",
        verify: Union[bool, str] = "/usr/share/ca-certificates/extra/ca.dsta.ai.crt",
    ) -> None:

        """
        Parameters
        ----------
        bucket : str
            Main S3 Bucket to link to
        endpoint_url : str
            Endpoint URL to private S3 storage
        aws_access_key_id : str
            S3 Access Key (Ask your SA for the key)
        aws_secret_access_key :  str
            S3 Secret Access Key (Ask your SA for the secret)
        signature_version : str
            Version of signature used for S3 Storage (default = 's3v4')
        region_name : str
            Region where the S3 storage is situated (default = 'us-east-1')
        verify : bool | str
            Verify with SSL
            if boolean: False - Do not verify
            if boolean: True - checks the environment variable REQUESTS_CA_BUNDLE for SSL Cert path
            Otherwise, insert SSL Cert path manually as string
        """

        self.bucket = bucket
        if isinstance(verify, str) and not os.path.exists(verify):
            print(f"{verify} does not exist. Downloading from relevant repository...")

            self.cert_output_dir = "/usr/share/ca-certificates/extra"
            os.makedirs(self.cert_output_dir, exist_ok=True)

            self.cert_download_url = (
                "https://gitlab.dsta.ai/ai-platform/getting-started/raw/master/config/ca.dsta.ai.crt"
            )
            req = requests.get(self.cert_download_url)

            verify = os.path.join(self.cert_output_dir, "ca.dsta.ai.crt")
            open(verify, "wb").write(req.content)
            print(f"{self.cert_download_url} has been downloaded to {verify}")

        self.s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version=signature_version),
            region_name=region_name,
            verify=verify,
        )

    def _get_data_path(self, bucket_name, folder, filename, local_folder):
        print(f"bucket_name: {bucket_name}, folder: {folder}, filename: {filename}, local_folder: {local_folder}")
        s3_path = folder + "/" + filename
        filename = filename.replace("/", "_")
        local_path = local_folder + "/" + filename
        return s3_path, local_path, filename

    def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket: str = None,
        verbose: int = 1,
    ) -> None:

        """
        Downloads a file from the S3 Storage to a local path.

        Parameters
        ----------
        s3_key : str
            Key of the file in the target bucket of the S3 Storage
        local_path : str
            Local path where the file will be downloaded to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [0: Print exceptions, 1: Print all] (default = 1)
        """

        if bucket is None:
            bucket = self.bucket

        if verbose == 1:
            print(f"S3 - Downloading from s3://{bucket}/{s3_key} to {local_path}")

        attempt_no = 1
        success = False

        while attempt_no <= self.MAX_ATTEMPT and (not success):
            try:
                print(f"downloading: {s3_key} to local_path: {local_path}")
                self.s3.Bucket(bucket).download_file(s3_key, local_path)
                success = True
                if verbose == 1:
                    print(f"S3 - File successfully downloaded to {local_path}")
            except ConnectionError as e:
                if attempt_no == self.MAX_ATTEMPT:
                    print("S3 - Max attempts reached. Connection to S3 not successful. Error is shown below.")
                    print(e)
                else:
                    print("S3 - connection failed. Retrying...")
                    time.sleep(self.WAIT_TIME)
                attempt_no += 1

        return success

    def download_zipped_file_v1(
        self, s3_key: str, local_path: str, bucket: str = None, temp_dir: str = "/tmp", remove_temp: bool = True
    ) -> None:

        """
        V1 Method - Extracts the contents of a S3 zipped file with 3 steps. (Time & Memory consuming, but stable)
        1. Downloads a zipped file from the S3 Storage to a temporary path.
        2. and then extracts the file into the target path.
        3. (Optional) Remove temporary zipped file.

        Parameters
        ----------
        s3_key : str
            Key of the file in the target bucket of the S3 Storage
        local_path : str
            Local path where the contents of the zipped file will be extracted to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        temp_dir : str
            Temporary directory where the zipped file will be stored (default = '/tmp')
        remove_temp : bool
            Boolean to remove temporary zip file (default = True)
        """

        # download file to temp directory
        temp_path = os.path.join(temp_dir, os.path.basename(s3_key))
        self.download_file(s3_key=s3_key, local_path=temp_path, bucket=bucket)

        # unzip file to target directory
        print(f"Local - Extracting contents from {temp_path} to {local_path}")
        with zipfile.ZipFile(temp_path, "r") as ref:
            ref.extractall(local_path)
        print(f"Local - File contents have been unzipped to {local_path}")

        # remove temporary zipped file
        if remove_temp:
            os.remove(temp_path)
            print(f"Local - {temp_path} has been deleted.")

        return None

    def download_zipped_file_v2(self, s3_key: str, local_path: str, bucket: str = None) -> None:

        """
        V2 Method - Reads and extracts the contents of a S3 zipped file as a bytes stream.
        (Only works for smaller zip files (<= 2GB), otherwise will overflow)

        Parameters
        ----------
        s3_key : str
            Key of the file in the target bucket of the S3 Storage
        local_path : str
            Local path where the contents of the zipped file will be extracted to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        """

        if bucket is None:
            bucket = self.bucket

        # instantiate variable
        zipped_object = None

        attempt_no = 1
        success = False

        while attempt_no <= self.MAX_ATTEMPT and (not success):
            try:
                zipped_object = self.s3.Object(bucket_name=bucket, key=s3_key)
                success = True
            except ConnectionError as e:
                if attempt_no == self.MAX_ATTEMPT:
                    print("S3 - Max attempts reached. Connection to S3 not successful. Error is shown below.")
                    print(e)
                    sys.exit(1)
                else:
                    print("S3 - connection failed. Retrying...")
                    time.sleep(self.WAIT_TIME)
                attempt_no += 1

        buffer = BytesIO(zipped_object.get()["Body"].read())
        ref = zipfile.ZipFile(buffer)
        for filename in ref.namelist():
            ref.extract(filename, local_path)

        print(f"Local - File contents from S3 have been extracted to {local_path}")
        return None

    def download_folder(
        self,
        s3_prefix: str,
        local_dir: str,
        bucket: str = None,
        verbose: int = 1,
    ) -> None:

        """
        Downloads all files from S3 Storage with the prefix to a local directory.

        Parameters
        ----------
        s3_prefix : str
            Prefix of the files in the target bucket of the S3 Storage
        local_dir : str
            Local directory where the files will be downloaded to
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [
                0: Only print exceptions,
                1: Only print main method logs,
                2: Print all
            ] (default = 1)
        """

        if bucket is None:
            bucket = self.bucket

        if verbose > 0:
            print(f"S3 - Downloading files with the prefix: {s3_prefix} to the local directory: {local_dir}")

        for obj in self.s3.Bucket(bucket).objects.filter(Prefix=s3_prefix):
            # skip directories
            if obj.key[-1] == "/":
                continue
            # path where each file will be locally downloaded to
            target_path = os.path.join(local_dir, os.path.relpath(obj.key, s3_prefix))
            # create nested directory if doesn't exist
            Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
            # download file
            sub_verbose = 0 if verbose <= 1 else 1
            self.download_file(s3_key=obj.key, local_path=target_path, bucket=bucket, verbose=sub_verbose)

        if verbose > 0:
            print(f"S3 - All files successfully downloaded to {local_dir}")

        return

    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        bucket: str = None,
        verbose: int = 1,
    ) -> None:

        """
        Upload a single file to S3

        Parameters
        ----------
        local_path : str
            Local path of the file for upload
        s3_key : str
            Key of the file to the target bucket of the S3 Storage
        bucket : str
            Manually state another bucket to upload to, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [0: Print exceptions, 1: Print all] (default = 1)
        """

        if bucket is None:
            bucket = self.bucket

        if verbose == 1:
            print(f"S3 - Uploading from {local_path} to s3://{bucket}/{s3_key}")

        attempt_no = 1
        success = False

        while attempt_no <= self.MAX_ATTEMPT and (not success):
            try:
                self.s3.Bucket(bucket).upload_file(local_path, s3_key)
                success = True
                if verbose == 1:
                    print(f"S3 - File has been successfully uploaded to s3://{bucket}/{s3_key}")

            except ConnectionError as e:
                if attempt_no == self.MAX_ATTEMPT:
                    print("S3 - Max attempts reached. Connection to S3 not successful. Error is shown below.")
                    print(e)
                    sys.exit(1)
                else:
                    print("S3 - connection failed. Retrying...")
                    time.sleep(self.WAIT_TIME)
                attempt_no += 1

        return None

    def upload_folder(
        self,
        local_dir: str,
        s3_prefix: str,
        bucket: str = None,
        verbose: int = 1,
    ) -> None:

        """
        Uploads all files from a local directory to S3 Storage with the inserted prefix.

        Parameters
        ----------
        local_dir : str
            Local directory where the files will be uploaded from
        s3_prefix : str
            Prefix of the files in the target bucket of the S3 Storage
        bucket : str
            Manually state another bucket to download from, otherwise download from main bucket (default = None)
        verbose : int
            Decide whether to print logs [
                0: Only print exceptions,
                1: Only print main method logs,
                2: Print all
            ] (default = 1)
        """

        if bucket is None:
            bucket = self.bucket

        if verbose > 0:
            print(f"S3 - Uploading files from the local directory: {local_dir} to S3 with prefix: {s3_prefix}")

        for dirx, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(dirx, file)
                s3_key = os.path.join(s3_prefix, os.path.relpath(local_path, local_dir))

                sub_verbose = 0 if verbose <= 1 else 1
                self.upload_file(local_path=local_path, s3_key=s3_key, bucket=bucket, verbose=sub_verbose)

        if verbose > 0:
            print(f"S3 - All files have been successfully uploaded to s3://{bucket}/{s3_prefix}")

        return
