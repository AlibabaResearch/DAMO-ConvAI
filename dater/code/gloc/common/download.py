import os
from tqdm import tqdm
import requests

def download_file(url, download_dir=None):
    """
    Download file into local file system from url
    """
    local_filename = url.split('/')[-1]
    if download_dir is None:
        download_dir = os.curdir
    elif not os.path.exists(download_dir):
        os.makedirs(download_dir)
    with requests.get(url, stream=True) as r:
        file_name = os.path.join(download_dir, local_filename)
        if os.path.exists(file_name):
            os.remove(file_name)
        write_f = open(file_name, "wb")
        for data in tqdm(r.iter_content()):
            write_f.write(data)
        write_f.close()

    return os.path.abspath(file_name)
