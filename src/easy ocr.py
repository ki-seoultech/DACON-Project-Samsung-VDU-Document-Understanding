# Colab 전용: EasyOCR 설치
!pip -q install easyocr==1.7.1

# 모델 가중치(ko+en) 미리 캐시 → 제출 번들에 동봉할 폴더에 저장
from pathlib import Path
import easyocr, torch, numpy as np

EASY_DIR = Path("/content/submit/model/easyocr")  # ← 제출 시 함께 압축할 위치
EASY_DIR.mkdir(parents=True, exist_ok=True)

# GPU 있으면 True, 없으면 False로 자동
use_gpu = torch.cuda.is_available()

# 첫 로딩은 download_enabled=True로 캐시 받기(Colab에서만!)
reader = easyocr.Reader(
    ['ko', 'en'],
    gpu=use_gpu,
    model_storage_directory=str(EASY_DIR),
    download_enabled=True,
    verbose=True
)

# 스모크 웜업 (가벼운 빈 이미지)
_ = reader.readtext(np.zeros((64, 256, 3), dtype=np.uint8))

# 어떤 파일이 내려왔는지 확인
for p in sorted(EASY_DIR.rglob("*")):
    if p.is_file():
        print("-", p)
from pathlib import Path
import easyocr, torch, numpy as np

EASY_DIR = Path("/content/submit/model/easyocr")
EASY_DIR.mkdir(parents=True, exist_ok=True)

# en만 따로 받아 두기
reader_en = easyocr.Reader(
    ['en'],
    gpu=torch.cuda.is_available(),
    model_storage_directory=str(EASY_DIR),
    download_enabled=True,   # ← Colab에서만 True로
    verbose=True
)
_ = reader_en.readtext(np.zeros((32,128,3), dtype=np.uint8))

# 확인
for p in sorted(EASY_DIR.rglob("*.pth")):
    print("-", p.name)

# 제출 루트/모델/입출력 폴더 생성
!mkdir -p  /content/submit/input /content/submit/output

from transformers import AutoTokenizer, AutoConfig, LayoutLMForSequenceClassification
import os

base_id = "microsoft/layoutlm-base-uncased"
out_dir = "/content/submit/model/layoutlm-base-uncased"
os.makedirs(out_dir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(base_id)
cfg = AutoConfig.from_pretrained(base_id, num_labels=2)  # 라벨수는 저장에 큰 영향X
model = LayoutLMForSequenceClassification.from_pretrained(base_id, config=cfg)

tok.save_pretrained(out_dir)
model.save_pretrained(out_dir)

# 확인
import os, glob
print("saved files:")
for p in sorted(glob.glob(out_dir+"/*")):
    print("-", os.path.basename(p))

%%writefile /content/submit/script.py
# /content/submit/script.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, subprocess, shutil, warnings, math
from pathlib import Path
import unicodedata

# ---- 오프라인/런 설정 ---------------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["ULTRALYTICS_HUB_ENABLED"] = "False"

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _fb = Path("/content/submit")
    SCRIPT_DIR = _fb if (_fb / "model" / "best.pt").exists() else Path.cwd()

INPUT_DIR  = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "submission.csv"

# ---- 모델 경로 ----------------------------------------------------------------
YOLO_WEIGHTS = SCRIPT_DIR / "model" / "best.pt"
EASY_DIR     = SCRIPT_DIR / "model" / "easyocr"
EASY_LANGS   = ['ko','en']

# 🔹 LayoutLM 베이스(오프라인)
BASE_DIR = SCRIPT_DIR / "model" / "layoutlm-base-uncased"

# 🔹 LayoutLM LoRA next-pair 위치 자동 탐색
PAIR_DIR_CANDIDATES = [
    Path("/content/runs/nextpair_lora"),
    SCRIPT_DIR / "model" / "nextpair_lora",
]
def _pick_pair_dir():
    for p in PAIR_DIR_CANDIDATES:
        if (p / "adapter_model.safetensors").exists() and (p / "adapter_config.json").exists():
            return p
    return None

# ---- 하이퍼파라미터 -----------------------------------------------------------
# Detector
IMG_SIZE=1280; DET_CONF=0.25; DET_IOU=0.7; MAX_DET=600
# 🔸 TTA+WBF
USE_TTA = True
TTA_IMGSZ = [1024, 1280, 1536]
TTA_ROTATIONS = [0, 90]  # 0°, 90°(clockwise)
WBF_IOU = 0.55
# 🔸 클래스별 post-filter 임계값
CLASS_THRESH = {"title":0.35, "subtitle":0.35, "text":0.25, "image":0.40, "table":0.45, "equation":0.40}

# OCR+리랭크
FUSE_W_DET=0.7; FUSE_W_OCR=0.3
SMALL_H_SKIP=12; BAND=32; COL_GAP_RATIO=0.08
TEXT_SET={"title","subtitle","text"}
CLASSES=["title","subtitle","text","image","table","equation"]

warnings.filterwarnings("ignore", category=DeprecationWarning)
import cv2, numpy as np, torch, pandas as pd
from ultralytics import YOLO

try:
    import fitz; _HAVE_PYMUPDF=True
except Exception:
    _HAVE_PYMUPDF=False

# ---- 유틸 ---------------------------------------------------------------------
def has_soffice(): return shutil.which("soffice") is not None

def pdf_to_images(pdf_path: Path):
    if not _HAVE_PYMUPDF: return []
    imgs=[]
    try:
        doc=fitz.open(str(pdf_path))
        for i in range(len(doc)):
            pix=doc.load_page(i).get_pixmap(dpi=240)
            img=np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
            imgs.append(cv2.imdecode(img, cv2.IMREAD_COLOR))
        doc.close()
    except Exception: return []
    return imgs

CONVERT_CACHE = SCRIPT_DIR / "work" / "converted"; CONVERT_CACHE.mkdir(parents=True, exist_ok=True)
def pptx_to_pdf(pptx_path: Path):
    if not has_soffice(): return None
    pdf_path = CONVERT_CACHE / (pptx_path.stem + ".pdf")
    if not pdf_path.exists():
        subprocess.run(["soffice","--headless","--convert-to","pdf","--outdir", str(CONVERT_CACHE), str(pptx_path)], check=True)
    return pdf_path

def file_to_images(path: Path):
    ext=path.suffix.lower()
    try:
        if ext in {".png",".jpg",".jpeg"}:
            im=cv2.imread(str(path)); return [im] if im is not None else []
        if ext==".pdf": return pdf_to_images(path)
        if ext in {".pptx",".ppt"}:
            pdf=pptx_to_pdf(path); return pdf_to_images(pdf) if pdf else []
    except Exception: return []
    return []

def enumerate_inputs(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            im = cv2.imread(str(p))
            if im is not None: yield p.name, 0, im
        elif ext == ".pdf":
            imgs = pdf_to_images(p)
            for i, img in enumerate(imgs):
                if img is not None: yield p.name, i, img
        elif ext in {".pptx", ".ppt"}:
            pdf = pptx_to_pdf(p)
            if pdf:
                imgs = pdf_to_images(pdf)
                for i, img in enumerate(imgs):
                    if img is not None: yield p.name, i, img

def normalize_text(s:str)->str:
    if not s: return ""
    s=unicodedata.normalize("NFKC", s).replace("\u00AD","").replace("\ufeff","")
    return " ".join(s.split())

def _parse_easyocr_output(pred):
    if pred and isinstance(pred[0], (list,tuple)) and len(pred[0])>=3 and not isinstance(pred[0][1], (list,tuple,dict)):
        txts=[str(t[1]) for t in pred]; confs=[float(t[2]) for t in pred]
        return " ".join(" ".join(txts).replace("\u00AD","").split()), (float(np.mean(confs)) if confs else 0.0)
    if pred and isinstance(pred[0], str):
        return " ".join(" ".join(pred).replace("\u00AD","").split()), 0.5
    return "", 0.0

# ---- 모델 로딩 ----------------------------------------------------------------
def load_models():
    assert YOLO_WEIGHTS.exists(), f"YOLO 가중치 없음: {YOLO_WEIGHTS}"
    det_model=YOLO(str(YOLO_WEIGHTS))
    try:
        import easyocr
        gpu_ok=torch.cuda.is_available()
        reader=easyocr.Reader(EASY_LANGS, gpu=gpu_ok, model_storage_directory=str(EASY_DIR), download_enabled=False, verbose=False)
        _=reader.readtext(np.zeros((48,192,3), dtype=np.uint8), detail=1, paragraph=True)
        print(f"[OK] EasyOCR ready (gpu={gpu_ok}) | dir={EASY_DIR}")
    except Exception as e:
        try: listing="\n".join(sorted(p.name for p in EASY_DIR.iterdir()))
        except Exception: listing="(list failed)"
        raise RuntimeError(f"EasyOCR 로드 실패. 가중치 동봉 필요.\n- 예상 위치: {EASY_DIR}\n- 기대 파일: craft_mlt_25k.pth, korean_g2.pth (english_g2.pth 선택)\n[DIR]\n{listing}\n[EXC] {e}")
    return det_model, reader

# ===================== TTA + WBF for YOLO =====================================
def _yolo_predict_xyxy(det_model, img_bgr, imgsz):
    res = det_model.predict(
        source=img_bgr, imgsz=imgsz, conf=DET_CONF, iou=DET_IOU, max_det=MAX_DET,
        device=0 if torch.cuda.is_available() else "cpu", half=torch.cuda.is_available(),
        save=False, verbose=False
    )[0]
    xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
    cls  = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,),dtype=int)
    conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
    return xyxy, cls, conf

def _rotate90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def _map_rot90_back(box_xyxy, orig_W, orig_H):
    # 입력: 회전(90° CW) 이미지 좌표의 [x1,y1,x2,y2], 회전된 이미지 크기 = (orig_H, orig_W)
    x1,y1,x2,y2 = map(float, box_xyxy)
    pts = [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
    back = []
    for x,y in pts:
        xo = y
        yo = orig_H - 1 - x
        back.append((xo,yo))
    xs = [p[0] for p in back]; ys = [p[1] for p in back]
    x1o, y1o, x2o, y2o = min(xs), min(ys), max(xs), max(ys)
    # 클립
    x1o = max(0, min(orig_W-1, x1o)); x2o = max(0, min(orig_W-1, x2o))
    y1o = max(0, min(orig_H-1, y1o)); y2o = max(0, min(orig_H-1, y2o))
    return [x1o, y1o, x2o, y2o]

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2,bx2) - max(ax1,bx1))
    ih = max(0.0, min(ay2,by2) - max(ay1,by1))
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = max(0.0,(ax2-ax1))*max(0.0,(ay2-ay1))
    area_b = max(0.0,(bx2-bx1))*max(0.0,(by2-by1))
    union = area_a + area_b - inter + 1e-6
    return inter/union

def _wbf_single_class(boxes, scores, iou_thr=WBF_IOU):
    # boxes: [[x1,y1,x2,y2]], scores: [float]
    picked = []
    for b, s in sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True):
        merged = False
        for k in range(len(picked)):
            pb, ps = picked[k]
            if _iou(b, pb) >= iou_thr:
                # 가중 평균
                w = ps + s + 1e-9
                nb = [
                    (pb[0]*ps + b[0]*s)/w,
                    (pb[1]*ps + b[1]*s)/w,
                    (pb[2]*ps + b[2]*s)/w,
                    (pb[3]*ps + b[3]*s)/w,
                ]
                ns = max(ps, s)  # 점수는 max로 보수적으로
                picked[k] = (nb, ns)
                merged = True
                break
        if not merged:
            picked.append((b, s))
    out_boxes  = [b for b,_ in picked]
    out_scores = [s for _,s in picked]
    return out_boxes, out_scores

def yolo_detect(det_model, img_bgr):
    H, W = img_bgr.shape[:2]
    # TTA 조합 수 줄이고 싶으면 아래 리스트 축소
    tta_specs = []
    for r in TTA_ROTATIONS:
        for sz in TTA_IMGSZ:
            tta_specs.append((r, sz))
    if not USE_TTA:
        tta_specs = [(0, IMG_SIZE)]

    # 수집 버퍼
    per_class_boxes = {c:[] for c in range(len(CLASSES))}
    per_class_scores = {c:[] for c in range(len(CLASSES))}

    for rot, sz in tta_specs:
        if rot == 0:
            img = img_bgr
            xyxy, cls, conf = _yolo_predict_xyxy(det_model, img, sz)
            # 좌표 바로 사용
            for b,c,s in zip(xyxy, cls, conf):
                per_class_boxes[c].append(b.tolist()); per_class_scores[c].append(float(s))
        elif rot == 90:
            img = _rotate90(img_bgr)  # shape: (W,H,3)
            xyxy_r, cls, conf = _yolo_predict_xyxy(det_model, img, sz)
            # 원본 좌표로 역변환
            for b,c,s in zip(xyxy_r, cls, conf):
                back = _map_rot90_back(b, W, H)
                per_class_boxes[c].append(back); per_class_scores[c].append(float(s))

    # 클래스별 WBF → post-filter → 최종 합치기
    outs = []
    for c in range(len(CLASSES)):
        boxes = per_class_boxes[c]; scores = per_class_scores[c]
        if not boxes: continue
        fboxes, fscores = _wbf_single_class(boxes, scores, iou_thr=WBF_IOU)
        # post-filter by class-specific threshold
        thr = CLASS_THRESH.get(CLASSES[c], DET_CONF)
        for b, s in zip(fboxes, fscores):
            if s < thr:
                continue
            x1,y1,x2,y2 = [int(round(v)) for v in b]
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            outs.append({"cls": CLASSES[c], "conf_det": float(s), "bbox":[x1, y1, x2-x1, y2-y1]})

    # 신뢰도 정렬 & max_det 컷
    outs.sort(key=lambda d: d["conf_det"], reverse=True)
    if len(outs) > MAX_DET:
        outs = outs[:MAX_DET]
    return outs
# ==============================================================================

# ---- OCR & 스코어 리랭크(E) ---------------------------------------------------
def fuse_score(d):
    base = d["conf_det"]
    if d["cls"] in TEXT_SET:
        o = d.get("conf_ocr", 0.0)
        text = d.get("text","") or ""
        L = len(text)
        boost = 0.0
        if L >= 12: boost += 0.05        # 적당히 긴 텍스트 보너스
        elif L <= 2: boost -= 0.05       # 지나치게 짧은 텍스트 페널티 (예: "-", ".")
        base = 0.7*base + 0.3*o + boost
    return float(np.clip(base, 0.0, 1.0))

def _prep_for_ocr(crop):
    pad = max(2, int(0.03 * max(crop.shape[:2])))
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thr = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def run_ocr(reader, img_bgr, dets):
    allow = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-–—_.,:;!?()[]{}@#%&+*/=~'\"··•°°℃% " \
            "가-힣ㄱ-ㅎㅏ-ㅣㆍ·,.;:!?()[]{}~‘’“”―‐-‐–—…· "
    for d in dets:
        if d["cls"] not in TEXT_SET:
            d.update(text="", conf_ocr=0.0); continue
        x, y, w, h = d["bbox"]
        if h < SMALL_H_SKIP:
            d.update(text="", conf_ocr=0.0); continue
        crop = img_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            d.update(text="", conf_ocr=0.0); continue
        min_side = max(1, min(crop.shape[:2]))
        scale = max(1.5, 64.0 / min_side)
        crop = cv2.resize(crop, (max(1,int(crop.shape[1]*scale)), max(1,int(crop.shape[0]*scale))))
        crop = _prep_for_ocr(crop)
        try:
            pred = reader.readtext(crop, detail=1, paragraph=False, decoder='beamsearch', allowlist=allow)
        except TypeError:
            pred = reader.readtext(crop, detail=1, paragraph=False)
        txts, confs = [], []
        for it in pred:
            if isinstance(it, (list, tuple)) and len(it) >= 3:
                t = str(it[1]).strip()
                c = float(it[2])
                if t: txts.append(t); confs.append(c)
        text = " ".join(" ".join(txts).replace("\u00AD","").split())
        conf = float(np.mean(confs)) if confs else 0.0
        d.update(text=normalize_text(text), conf_ocr=float(np.clip(conf, 0.0, 1.0)))
    return dets

# ---- 지오메트리 기반 1차 순서 --------------------------------------------------
def geo_topo_order(dets, page_w, page_h):
    items=[]
    for i,d in enumerate(dets):
        x,y,w,h=d["bbox"]; xc=x+w/2; items.append((i,x,y,w,h,xc))
    if not items: return []
    items.sort(key=lambda t: t[5])
    cols,cur=[],[items[0]]; gap_thr=page_w*COL_GAP_RATIO
    for a,b in zip(items, items[1:]):
        cur.append(b)
        if b[5]-a[5]>gap_thr:
            cols.append(cur[:-1]); cur=[b]
    cols.append(cur)
    def band_key(i): x,y,w,h=dets[i]["bbox"]; return (int(y//BAND), x)
    order=[]
    for col in cols:
        ids=[t[0] for t in col]; ids.sort(key=band_key); order.extend(ids)
    return order

# ---- next-pair LoRA 로더 ------------------------------------------------------
def try_load_pair_model():
    pair_dir = _pick_pair_dir()
    if pair_dir is None:
        print("[info] next-pair LoRA not found -> geo-only reading order.")
        return None, None, None
    try:
        from transformers import AutoTokenizer, AutoConfig, LayoutLMForSequenceClassification
        from peft import PeftModel
        base = str(BASE_DIR) if BASE_DIR.exists() else "microsoft/layoutlm-base-uncased"

        tok  = AutoTokenizer.from_pretrained(base)
        cfg  = AutoConfig.from_pretrained(base, num_labels=2)
        base_model = LayoutLMForSequenceClassification.from_pretrained(base, config=cfg)

        model = PeftModel.from_pretrained(base_model, str(pair_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        print(f"[OK] next-pair loaded from {pair_dir} (device={device}) | base={'local' if BASE_DIR.exists() else 'hf'}")
        return tok, model, device
    except Exception as e:
        print(f"[warn] next-pair load failed -> geo-only. ({e})")
        return None, None, None

# ---- pair 인퍼런스 인코딩 -----------------------------------------------------
def _encode_pair_infer(tok, text_a, box_a, text_b, box_b, W, H, max_len=512):
    def norm_box_xyxy(x,y,w,h):
        x1,y1,x2,y2 = x, y, x+w, y+h
        nx1 = int(1000 * max(0, min(1, x1/float(W))))
        ny1 = int(1000 * max(0, min(1, y1/float(H))))
        nx2 = int(1000 * max(0, min(1, x2/float(W))))
        ny2 = int(1000 * max(0, min(1, y2/float(H))))
        return [nx1, ny1, nx2, ny2]
    ba = norm_box_xyxy(*box_a)
    bb = norm_box_xyxy(*box_b)
    enc_a = tok(text_a or "", add_special_tokens=False)
    enc_b = tok(text_b or "", add_special_tokens=False)
    room = max_len - 3
    la = min(len(enc_a["input_ids"]), room//2)
    lb = min(len(enc_b["input_ids"]), room - la)
    ids  = [tok.cls_token_id] + enc_a["input_ids"][:la] + [tok.sep_token_id] + enc_b["input_ids"][:lb] + [tok.sep_token_id]
    attn = [1]*len(ids)
    ttid = [0]*(1+la+1) + [1]*(lb+1)
    bbox = [[0,0,0,0]] + [ba]*la + [[0,0,0,0]] + [bb]*lb + [[0,0,0,0]]
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.tensor([attn]),
        "token_type_ids": torch.tensor([ttid]),
        "bbox": torch.tensor([bbox]),
    }

# ---- (참고) 로컬 스왑 보정: 유지만, 현재는 미사용 ------------------------------
def refine_order_with_pair(dets, order, tok, model, device, W, H, passes=2, margin=0.05):
    if tok is None or model is None or device is None or not order:
        return order
    ord_list = order[:]
    ord_list = [i for i in ord_list if dets[i]["cls"] in TEXT_SET]
    if len(ord_list) < 2: return ord_list
    with torch.no_grad():
        for _ in range(passes):
            swapped = False
            for k in range(len(ord_list)-1):
                ia, ib = ord_list[k], ord_list[k+1]
                A, B = dets[ia], dets[ib]
                xa,ya,wa,ha = A["bbox"]; xb,yb,wb,hb = B["bbox"]
                batch_ab = _encode_pair_infer(tok, A.get("text",""), (xa,ya,wa,ha),
                                                   B.get("text",""), (xb,yb,wb,hb), W, H)
                batch_ba = _encode_pair_infer(tok, B.get("text",""), (xb,yb,wb,hb),
                                                   A.get("text",""), (xa,ya,wa,ha), W, H)
                batch_ab = {k:v.to(device) for k,v in batch_ab.items()}
                batch_ba = {k:v.to(device) for k,v in batch_ba.items()}
                logits_ab = model(**batch_ab).logits.softmax(-1)[0]
                logits_ba = model(**batch_ba).logits.softmax(-1)[0]
                p_ab = float(logits_ab[1].item())
                p_ba = float(logits_ba[1].item())
                if p_ba > p_ab + margin:
                    ord_list[k], ord_list[k+1] = ord_list[k+1], ord_list[k]
                    swapped = True
            if not swapped:
                break
    return ord_list

# ---- 전역 빔서치 리오더 -------------------------------------------------------
@torch.inference_mode()
def _pair_prob(tok, model, device, W, H, A_text, A_box, B_text, B_box):
    batch = _encode_pair_infer(tok, A_text, A_box, B_text, B_box, W, H)
    out = model(**{k:v.to(device) for k,v in batch.items()}).logits
    p = torch.softmax(out, -1)[0, 1].item()
    return max(1e-6, min(1-1e-6, p))

def _geo_prior(last_box, next_box):
    lx, ly, lw, lh = last_box
    nx, ny, nw, nh = next_box
    same_col = abs((lx+lw/2) - (nx+nw/2)) <= 0.6 * max(lw, nw)
    dy = (ny - ly) / (lh + nh + 1e-3)
    score = (0.6 if same_col else 0.0) + 0.4 * max(0.0, dy) - 0.1 * max(0.0, -dy)
    return score

def reorder_with_pair_beam(dets, indices, tok, model, device, W, H, beam=8, alpha_geo=0.2):
    if not indices or tok is None or model is None:
        return indices
    starts = sorted(indices, key=lambda i: dets[i]["bbox"][1])[:min(len(indices), beam)]
    sequences = [([s], 0.0) for s in starts]

    for _ in range(1, len(indices)):
        new_seqs = []
        for seq, score in sequences:
            used = set(seq)
            last = seq[-1]
            for j in indices:
                if j in used:
                    continue
                A = dets[last]; B = dets[j]
                p = _pair_prob(tok, model, device, W, H, A.get("text",""), A["bbox"], B.get("text",""), B["bbox"])
                geo = _geo_prior(A["bbox"], B["bbox"])
                new_score = score + math.log(p) + alpha_geo * geo
                new_seqs.append((seq+[j], new_score))
        new_seqs.sort(key=lambda x: x[1], reverse=True)
        sequences = new_seqs[:beam]

    best_seq = max(sequences, key=lambda x: x[1])[0] if sequences else indices
    return best_seq

# ---- 파이프라인 ---------------------------------------------------------------
def process_image_core(file_name, page_idx, img_bgr, det_model, reader, pair_tok=None, pair_model=None, pair_device=None):
    H,W=img_bgr.shape[:2]
    dets=yolo_detect(det_model, img_bgr)          # ← TTA+WBF 적용
    dets=run_ocr(reader, img_bgr, dets)
    for d in dets: d["score"]=fuse_score(d)

    all_geo = geo_topo_order(dets, W, H)
    text_geo = [i for i in all_geo if dets[i]["cls"] in TEXT_SET]
    non_text_geo = [i for i in all_geo if dets[i]["cls"] not in TEXT_SET]

    if pair_tok and pair_model:
        text_fused = reorder_with_pair_beam(dets, text_geo, pair_tok, pair_model, pair_device, W, H, beam=8, alpha_geo=0.2)
    else:
        text_fused = text_geo

    final_order = text_fused + non_text_geo

    rows=[]
    for rank,idx in enumerate(final_order, start=1):
        d=dets[idx]; x,y,w,h=d["bbox"]
        rows.append([file_name, page_idx, d["cls"], d.get("text",""), rank, x, y, w, h, f"{d['score']:.5f}"])
    return rows

def main():
    print(f"[paths] SCRIPT_DIR={SCRIPT_DIR}")
    print(f"[paths] YOLO_WEIGHTS={YOLO_WEIGHTS}")
    print(f"[paths] EASY_DIR    ={EASY_DIR}")
    if BASE_DIR.exists(): print(f"[paths] BASE_DIR    ={BASE_DIR}")
    picked = _pick_pair_dir()
    print(f"[paths] NEXTPAIR_DIR={picked if picked else '(none)'}")
    print(f"[cfg] USE_TTA={USE_TTA} | TTA_ROTATIONS={TTA_ROTATIONS} | TTA_IMGSZ={TTA_IMGSZ} | WBF_IOU={WBF_IOU}")

    det, reader = load_models()
    pair_tok, pair_model, pair_device = try_load_pair_model()

    header=["ID","category_type","confidence_score","order","text","bbox"]
    server_csv=SCRIPT_DIR/"data"/"test.csv"
    server_mode=server_csv.exists()

    if server_mode:
        df = pd.read_csv(server_csv)
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(header)
            for row in df.itertuples(index=False):
                doc_id = str(getattr(row, "ID"))
                raw = str(getattr(row, "path")).strip()
                src = (SCRIPT_DIR / raw) if raw.startswith("data/") else (SCRIPT_DIR / "data" / raw)
                imgs = file_to_images(src)
                if not imgs: continue
                rows=process_image_core(src.name,0,imgs[0],det,reader,pair_tok,pair_model,pair_device)
                for r in rows:
                    _,_,category,text,order,x,y,w,h,score=r
                    x1,y1,x2,y2=int(x),int(y),int(x+w),int(y+h)
                    wr.writerow([doc_id,category,f"{float(score):.6f}",int(order),text,f"{x1},{y1},{x2},{y2}"])
        print(f"[OK] submission.csv -> {OUTPUT_CSV.resolve()}")
    else:
        if not INPUT_DIR.exists():
            raise AssertionError(f"입력 폴더가 없습니다: {INPUT_DIR}")
        with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as f:
            wr=csv.writer(f); wr.writerow(header)
            for fname,page_idx,img in enumerate_inputs(INPUT_DIR):
                rows=process_image_core(fname,page_idx,img,det,reader,pair_tok,pair_model,pair_device)
                for r in rows:
                    _,_,category,text,order,x,y,w,h,score=r
                    x1,y1,x2,y2=int(x),int(y),int(x+w),int(y+h)
                    wr.writerow([Path(fname).stem,category,f"{float(score):.6f}",int(order),text,f"{x1},{y1},{x2},{y2}"])
        print(f"[OK] (local) submission.csv -> {OUTPUT_CSV.resolve()}")

if __name__=="__main__":
    main()

!mkdir -p /content/submit/model/nextpair_lora
!cp -r /content/runs/nextpair_lora/* /content/submit/model/nextpair_lora/

%%bash
set -e
cd /content/submit

# 1) requirements.txt: LoRA용 peft만 최소 명시
printf "peft==0.11.1\n" > requirements.txt

# 2) 이전 zip 삭제 후 새로 압축
rm -f ../submit.zip
zip -r ../submit.zip script.py requirements.txt model \
  -x 'input/*' 'output/*' 'data/*' 'work/*' \
     '**/__pycache__/*' '**/.ipynb_checkpoints/*' 'runs/*' '.git/*'

# 3) 구조 확인 (submit/ 프리픽스가 없어야 정상)
unzip -l ../submit.zip | sed -n '1,120p'
