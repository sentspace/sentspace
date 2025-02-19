import importlib
import boto3
import os
import sys
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from pathlib import Path
import subprocess


class _LinkStorage:
    """
    load pickles using a download link
    """

    def __init__(
        self, script_path=Path(__file__).parent / "download_norms.sh", *args, **kwargs
    ):
        """
        executes a download script
        """
        self._script_path = script_path
        if os.path.isfile(self._script_path):
            subprocess.call(["bash", self._script_path])
        else:
            raise FileNotFoundError(f"The script {self._script_path} does not exist.")


class _S3Storage:
    """
    load pickles
    """

    _NO_SIGNATURE = Config(signature_version=UNSIGNED)

    def __init__(
        self,
        *args,
        key,
        bucket="sentspace-databases",
        region="us-east-1",
        root_dir=os.getcwd() + "/.feature_database/",
        **kwargs,
    ):
        super(_S3Storage, self).__init__(*args, **kwargs)
        self._key = key
        self._bucket = bucket
        self._region = region
        # self._local_root_dir = os.path.join(os.getcwd(),'.feature_database/')
        self._local_root_dir = str(Path(root_dir).expanduser().resolve()) + "/"
        os.makedirs(self._local_root_dir, exist_ok=True)
        self._retrieve()

    # is the relevant file here?
    def _retrieve(self):
        key = self._key
        dir = self._local_root_dir
        local_path = os.path.join(dir, key)
        if not os.path.isfile(local_path):
            self._download_file(key, local_path)

    # download it from s3 bucket
    def _download_file(self, key, local_path):
        print(f"Downloading {key} to {local_path}")
        s3 = boto3.resource("s3", region_name=self._region, config=self._NO_SIGNATURE)
        obj = s3.Object(self._bucket, key)
        with tqdm(
            total=obj.content_length,
            unit="B",
            unit_scale=True,
            desc=key,
            file=sys.stdout,
        ) as progress_bar:

            def progress_hook(bytes_amount):
                progress_bar.update(bytes_amount)

            obj.download_file(local_path, Callback=progress_hook)

    def save(self, result, function_identifier):
        raise NotImplementedError("can only load from S3, but not save")


# load_feature = _S3Storage
load_feature = _LinkStorage
