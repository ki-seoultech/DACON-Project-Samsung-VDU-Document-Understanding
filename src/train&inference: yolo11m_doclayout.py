# yolo11m.pt 저장코드
# ===== Safe training to Drive: 11m@1024, 20ep, patience=10, scan-friendly, save_period=1 =====
import sys, subprocess, os, gc, shutil, time
from pathlib import Path

def ensure_pkg(p):
    try: __import__(p.split("==")[0].split(">=")[0].split("[")[0])
    except Exception: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

ensure_pkg("ultralytics>=8.3.0")
import torch
from ultralytics import YOLO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
try: torch.cuda.empty_cache()
except: pass
gc.collect()

# ---- Paths (Drive에 저장) ----
DATA_YAML = "/content/workspace/dataset_balanced/data.yaml"
MODEL = "yolo11m.pt"
PROJECT = "/content/drive/MyDrive/runs_doc"      # ★ Drive 경로
SAVE_ROOT = "/content/drive/MyDrive/submit_model" # ★ 최종 복사 경로도 Drive
Path(PROJECT).mkdir(parents=True, exist_ok=True)
Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

# ---- 공통 설정 (스캔 친화 증강) ----
COMMON = dict(
    data=DATA_YAML,
    epochs=20,
    device=0,
    workers=2,
    seed=42,
    project=PROJECT,
    cache=False,          # 디스크 캐시 용량 경고 회피
    patience=10,
    save_period=1,        # ★ 매 에폭 checkpoint 저장
    exist_ok=True,

    optimizer="AdamW",
    lr0=0.003, lrf=0.12, cos_lr=True,
    weight_decay=5e-4,
    warmup_epochs=3,

    hsv_h=0.0, hsv_s=0.0, hsv_v=0.05,
    degrees=0.5, translate=0.02, scale=0.5,
    shear=0.0, perspective=0.0002,
    fliplr=0.0, flipud=0.0,
    mosaic=0.03, mixup=0.0, copy_paste=0.0, erasing=0.0,
    close_mosaic=6,

    box=8.0, cls=0.5, dfl=1.5,
    amp=True,
)

# ---- 실행 (타임스탬프 run name으로 충돌 방지) ----
run_name = f"doc11m_1024_{time.strftime('%m%d_%H%M%S')}"
print(f"\n==> TRAIN: imgsz=1024, batch=-1(auto), run={run_name}")
model = YOLO(MODEL)
res = model.train(imgsz=1024, batch=-1, name=run_name, **COMMON)

# ---- best/last를 Drive에 복사 ----
run_dir = Path(PROJECT)/"detect"/run_name
wdir = run_dir/"weights"
best, last = wdir/"best.pt", wdir/"last.pt"
ts = time.strftime("%Y%m%d-%H%M%S")
for src, canon in [(best,"best.pt"), (last,"last.pt")]:
    shutil.copy2(src, Path(SAVE_ROOT)/canon)
    shutil.copy2(src, Path(SAVE_ROOT)/f"{canon.split('.')[0]}_{ts}.pt")
print("✅ saved to", SAVE_ROOT)

#이어서 가중치학습
from ultralytics import YOLO
RUN = "/content/drive/MyDrive/runs_doc/doc11m_1024_0902_052232"
DATA_YAML = "/content/workspace/dataset_balanced/data.yaml"

m = YOLO(f"{RUN}/weights/last.pt")
m.train(resume=True, data=DATA_YAML,
        project="/content/drive/MyDrive/runs_doc",
        name="doc11m_1024_0902_052232",
        workers=2)

# best.pt가져오기!!!
from google.colab import drive; drive.mount('/content/drive')
%cd /content
!mkdir -p submit/model

# 2) Drive에 저장된 best.pt 경로 지정
BEST = "/content/drive/MyDrive/runs_doc/doc11m_1024_0902_052232/weights/best.pt"

# 3) 제출 폴더로 복사(이름은 기존 파이프라인 기대대로 best.pt)
import shutil, os
shutil.copy2(BEST, "/content/submit/model/best.pt")
