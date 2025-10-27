!pip -q install huggingface_hub datasets kaggle pycocotools pillow lxml tqdm
import os, json, pathlib, shutil
from huggingface_hub import snapshot_download
base = pathlib.Path("/content")
(base/"datasets").mkdir(parents=True, exist_ok=True)
(base/"workspace").mkdir(parents=True, exist_ok=True)

from google.colab import files
print("Kaggle API 토큰(kaggle.json) 업로드 창이 뜹니다.")
files.upload();  # kaggle.json 선택
!mkdir -p /root/.kaggle && mv kaggle.json /root/.kaggle/ && chmod 600 /root/.kaggle/kaggle.json
!kaggle -v
print("✅ Kaggle 토큰 설정 완료")

# DocLayNet core (COCO/PNG 포함)
!mkdir -p /content/datasets/doclaynet && cd /content/datasets/doclaynet
!wget -q --show-progress "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
!unzip -q DocLayNet_core.zip -d .
!ls -R | head -n 60

