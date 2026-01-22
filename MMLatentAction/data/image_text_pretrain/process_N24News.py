import zipfile
import os

zip_path = "N24News.zip"
extract_to = "N24News"


os.makedirs(extract_to, exist_ok=True)

# 解压
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
