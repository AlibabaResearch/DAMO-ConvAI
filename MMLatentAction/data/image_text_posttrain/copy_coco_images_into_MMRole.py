

import zipfile
import os

def unzip_file(zip_path, extract_to):
    # 确保目标文件夹存在
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# 使用示例
zip_file = "manh6054/MSCOCO/train2017.zip"
output_folder = "YanqiDai/MMRole_dataset/images/COCO/train2017"

unzip_file(zip_file, output_folder)
