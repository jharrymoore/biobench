import tempfile
from urllib.request import urlretrieve
import zipfile


class Experiment:
    def __init__(
        self,
        data_url: str,
        data_path: str,
        partition: str,
        timelimit: str,
        account: str,
        overwrite: bool,
    ):
        self.data_url = data_url
        self.data_path = data_path
        self.partition = partition
        self.timelimit = timelimit
        self.account = account
        self.overwrite = overwrite

    def download_data(self, tmp_path: str):
        # download data from url
        s = urlretrieve(self.data_url, tmp_path)

    def unzip_data(self, tmp_path: str):
        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            zip_ref.extractall(self.data_path)

    def prepare_data(self):
        tmp_path = tempfile.mkdtemp()
        self.download_data(tmp_path)
        self.unzip_data(tmp_path)
