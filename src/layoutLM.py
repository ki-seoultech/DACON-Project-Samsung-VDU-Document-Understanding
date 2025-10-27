!mkdir -p tools
!mkdir -p data/pairs
!rm -f data/pages.jsonl

%%writefile tools/yolo_to_pages_jsonl.py
import os, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from statistics import median
import numpy as np

# 텍스트 클래스 (YOLO 라벨)
TEXT_CLASS_IDS = {0, 1, 2}  # title, subtitle, text

# -------------------------
# 유틸
# -------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w/2) * W;  y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W;  y2 = (yc + h/2) * H
    x1 = clamp(x1, 0, W-1); x2 = clamp(x2, 0, W-1)
    y1 = clamp(y1, 0, H-1); y2 = clamp(y2, 0, H-1)
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def word_in_block_center(wbox, bbox, margin=5):
    wx1, wy1, wx2, wy2 = wbox
    cx = (wx1 + wx2) / 2.0
    cy = (wy1 + wy2) / 2.0
    bx1, by1, bx2, by2 = bbox
    return (bx1 - margin) <= cx <= (bx2 + margin) and (by1 - margin) <= cy <= (by2 + margin)

def crop_with_padding(im: Image.Image, box, pad=10):
    W, H = im.size
    x1, y1, x2, y2 = box
    x1 = clamp(x1 - pad, 0, W-1)
    y1 = clamp(y1 - pad, 0, H-1)
    x2 = clamp(x2 + pad, 0, W-1)
    y2 = clamp(y2 + pad, 0, H-1)
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    return im.crop((x1, y1, x2, y2))

def only_digits_punct(s: str) -> bool:
    if not s: return False
    return all((c.isdigit() or c.isspace() or c in "-–—_.,:;()[]{}%/\\'\"+&@#$*?!") for c in s)

def dehyphenate(text: str) -> str:
    import re
    # 줄바꿈 하이픈 복원 + 공백 정규화 + 짧은 쓰레기 토큰 제거
    t = re.sub(r'(\w)-\s*\n?\s*(\w)', r'\1\2', text)
    t = re.sub(r'\s+', ' ', t).strip()
    toks = [tok for tok in t.split() if (len(tok) > 2 or any(c.isalnum() for c in tok))]
    return " ".join(toks)

# -------------------------
# EasyOCR (온라인 허용: 필요 시 자동 다운로드)
# -------------------------
class EasyOCRBackend:
    def __init__(self, langs=("ko","en"), model_dir="model/easyocr", gpu=True):
        import torch
        from easyocr import Reader
        # 여기서는 사용자가 넘겨준 model_dir를 그대로 사용하고,
        # 가중치가 없으면 인터넷에서 자동으로 받아오도록 download_enabled=True
        self.reader = Reader(
            list(langs),
            gpu=(gpu and torch.cuda.is_available()),
            model_storage_directory=str(model_dir),
            download_enabled=True,    # ← 자동 다운로드 허용
            verbose=True
        )

    def read_words(self, img_or_arr,
                   height_ths=0.6, width_ths=0.6,
                   mag_ratio=1.5, low_text=0.3, link_threshold=0.3, text_threshold=0.5,
                   decoder='greedy'):
        """
        반환: List[(text, [x1,y1,x2,y2], conf)]
        """
        if isinstance(img_or_arr, Image.Image):
            arr = np.array(img_or_arr.convert("RGB"))
        elif isinstance(img_or_arr, (str, Path)):
            arr = str(img_or_arr)
        else:
            arr = img_or_arr  # numpy array 라고 가정

        try:
            res = self.reader.readtext(
                arr, detail=1, paragraph=False,
                height_ths=height_ths, width_ths=width_ths,
                mag_ratio=mag_ratio, low_text=low_text,
                link_threshold=link_threshold, text_threshold=text_threshold,
                decoder=decoder
            )
        except Exception:
            return []

        out = []
        for item in (res or []):
            try:
                pts, txt, conf = item
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                txt = (txt or "").strip()
                if txt:
                    out.append((txt, box, float(conf)))
            except Exception:
                pass
        return out

# -------------------------
# 읽기순서 v2
#  - 동적 band
#  - 스팬(폭 넓은 블록) 우선
#  - 열 인식(중심 간격 허들)
#  - 헤더/풋터/페이지넘버 후순위
# -------------------------
def geo_topo1_v2(blocks: List[Dict[str,Any]], W: int, H: int,
                 band_scale: float = 1.2, col_gap_ratio: float = 0.08, span_width_ratio: float = 0.65):
    if not blocks: return []

    # 동적 band
    hs = [(b["box"][3]-b["box"][1]) for b in blocks]
    band = max(24, int(round(median(hs) * band_scale))) if hs else 32

    # 열 인식
    items=[]
    for b in blocks:
        x1,y1,x2,y2=b["box"]; xc=(x1+x2)/2.0
        items.append((b["id"], x1, y1, x2, y2, xc))
    items.sort(key=lambda t: t[5])  # by x-center

    cols=[]; gap_thr=W*col_gap_ratio
    if items:
        cur=[items[0]]
        for a,b in zip(items, items[1:]):
            if b[5]-a[5] > gap_thr:
                cols.append(cur); cur=[b]
            else:
                cur.append(b)
        cols.append(cur)

    # 정렬 키 구성
    def is_header_footer(b):
        x1,y1,x2,y2 = b["box"]
        small_h = (y2-y1) <= max(14, int(0.02*H))
        top = y1 <= int(0.05*H)
        bottom = y2 >= int(0.95*H)
        txt = (b.get("text") or "").strip()
        short_txt = (len(txt) <= 6) or only_digits_punct(txt)
        return (small_h and (top or bottom)) or ((top or bottom) and short_txt)

    # span/열/행 정렬
    order=[]
    for ci, col in enumerate(cols):
        ids=[t[0] for t in col]
        # 미리 span 플래그 채우기
        for i in ids:
            x1,y1,x2,y2 = blocks[i]["box"]
            blocks[i]["is_span"] = ((x2-x1)/float(W)) >= span_width_ratio
            blocks[i]["col_idx"] = ci

        ids.sort(key=lambda i: (
            0 if blocks[i]["is_span"] else 1,                 # 스팬 우선
            1 if is_header_footer(blocks[i]) else 0,          # 헤더/풋터 뒤로
            int(blocks[i]["box"][1] // band),                 # 밴드 단위 Y 정렬
            blocks[i]["box"][0]                                # tie-break by x1
        ))
        order.extend(ids)
    return order

# -------------------------
# 메인
# -------------------------
def run(yolo_root: str, out_path: str, splits: List[str],
        easyocr_lang: str, easyocr_dir: str,
        max_pages: Optional[int],
        word_conf_thr: float,
        drop_weak: int, min_chars: int, min_conf: float,
        roi_ocr: int, roi_pad: int,
        band_scale: float, col_gap_ratio: float, span_width_ratio: float):

    yroot = Path(yolo_root)
    outp  = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # OCR 준비 (온라인 허용)
    langs = tuple([s.strip() for s in easyocr_lang.split(",") if s.strip()])
    ocr = EasyOCRBackend(langs=langs, model_dir=easyocr_dir)

    written = 0
    for split in splits:
        img_dir = yroot / "images" / split
        lab_dir = yroot / "labels" / split
        if not img_dir.exists() or not lab_dir.exists():
            print(f"[warn] missing split: {split}")
            continue

        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        for img in imgs:
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue

            # 이미지 사이즈
            try:
                with Image.open(img) as im:
                    W, H = im.size
            except Exception:
                continue

            # 블록 로드(텍스트 후보만)
            blocks=[]
            try:
                with open(lab, "r", encoding="utf-8") as f:
                    for ln in f:
                        p = ln.strip().split()
                        if len(p) < 5: continue
                        cid = int(float(p[0]))
                        if cid not in TEXT_CLASS_IDS: continue
                        xc, yc, w, h = map(float, p[1:5])
                        blocks.append({"id": len(blocks), "box": yolo_to_xyxy(xc, yc, w, h, W, H)})
            except Exception:
                continue

            if len(blocks) < 2:
                continue

            # 1) 전페이지 OCR
            words_full = ocr.read_words(
                str(img),
                mag_ratio=1.5, low_text=0.3, link_threshold=0.3, text_threshold=0.5,
                height_ths=0.6, width_ths=0.6, decoder='greedy'
            )

            # 2) 단어-블록 매칭 (center+margin, conf 필터)
            for b in blocks:
                wtexts=[]; wboxes=[]; wconfs=[]
                for (txt, wbox, conf) in words_full:
                    if conf < word_conf_thr:
                        continue
                    if word_in_block_center(wbox, b["box"], margin=5):
                        wtexts.append(txt); wboxes.append(wbox); wconfs.append(conf)
                b["text"] = dehyphenate(" ".join(wtexts)) if wtexts else ""
                b["word_boxes"] = wboxes if wboxes else []
                b["block_conf"] = float(median(wconfs)) if wconfs else 0.0

            # 3) ROI 재OCR (빈/약한 블록만)
            if roi_ocr:
                try:
                    im = Image.open(img).convert("RGB")
                except Exception:
                    im = None
                if im is not None:
                    for b in blocks:
                        need = (len(b.get("text","")) < max(1, min_chars))
                        if not need:
                            continue
                        roi = crop_with_padding(im, b["box"], pad=int(roi_pad))
                        words_roi = ocr.read_words(
                            roi,
                            mag_ratio=2.0, low_text=0.2, link_threshold=0.2, text_threshold=0.4,
                            height_ths=0.6, width_ths=0.6, decoder='greedy'
                        )
                        if words_roi:
                            wtexts=[w[0] for w in words_roi]
                            wboxes=[w[1] for w in words_roi]
                            wconfs=[w[2] for w in words_roi]
                            b["text"] = dehyphenate(" ".join(wtexts))
                            b["word_boxes"] = wboxes
                            b["block_conf"] = float(median(wconfs)) if wconfs else b.get("block_conf", 0.0)

            # 4) 읽기순서 v2
            order = geo_topo1_v2(
                blocks, W, H,
                band_scale=float(band_scale), col_gap_ratio=float(col_gap_ratio), span_width_ratio=float(span_width_ratio)
            )

            # 5) 노이즈 필터 (옵션)
            if drop_weak:
                kept=[]
                for b in blocks:
                    txt = (b.get("text") or "").strip()
                    conf_ok = (b.get("block_conf", 0.0) >= float(min_conf))
                    char_ok = (len(txt) >= int(min_chars))
                    if char_ok and conf_ok:
                        kept.append(b)
                kept_ids = {b["id"] for b in kept}
                order = [bid for bid in order if bid in kept_ids]
                blocks = kept

            # 6) 저장
            line = {
                "page_image": str(img),
                "page_size": [W, H],
                "blocks": blocks,
                "reading_order": order
            }
            with open(outp, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1

            if max_pages and written >= max_pages:
                print(f"[info] max_pages reached: {written}")
                break

    print(f"[OK] pages.jsonl written: {written} -> {out_path}")

# -------------------------
# CLI
# -------------------------
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-root", required=True)
    ap.add_argument("--out", default="data/pages.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train","val"])
    ap.add_argument("--easyocr-lang", default="ko,en")
    ap.add_argument("--easyocr-dir", default="model/easyocr")  # ← 온라인 캐시 디렉토리
    ap.add_argument("--max-pages", type=int, default=None)

    # 요청한 개선 파라미터들
    ap.add_argument("--word-conf-thr", type=float, default=0.30)
    ap.add_argument("--drop-weak", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=5)
    ap.add_argument("--min-conf", type=float, default=0.20)
    ap.add_argument("--roi-ocr", type=int, default=1)
    ap.add_argument("--roi-pad", type=int, default=10)
    ap.add_argument("--band-scale", type=float, default=1.2)
    ap.add_argument("--col-gap-ratio", type=float, default=0.08)
    ap.add_argument("--span-width-ratio", type=float, default=0.65)

    args = ap.parse_args()
    run(
        yolo_root=args.yolo_root,
        out_path=args.out,
        splits=args.splits,
        easyocr_lang=args.easyocr_lang,
        easyocr_dir=args.easyocr_dir,
        max_pages=args.max_pages,
        word_conf_thr=args.word_conf_thr,
        drop_weak=args.drop_weak,
        min_chars=args.min_chars,
        min_conf=args.min_conf,
        roi_ocr=args.roi_ocr,
        roi_pad=args.roi_pad,
        band_scale=args.band_scale,
        col_gap_ratio=args.col_gap_ratio,
        span_width_ratio=args.span_width_ratio,
    )

#layoutLM + loRA 데이터 준비단계


YOLO_ROOT = "/content/workspace/dataset_balanced"

!rm -f data/pages.jsonl
!python tools/yolo_to_pages_jsonl.py \
  --yolo-root "$YOLO_ROOT" \
  --out data/pages.jsonl \
  --splits train val \
  --easyocr-lang ko,en \
  --easyocr-dir model/easyocr \
  --max-pages 200

import os, itertools, json
print("exists?", os.path.exists("data/pages.jsonl"))
with open("data/pages.jsonl","r",encoding="utf-8") as f:
    for line in itertools.islice(f, 10):
        print(json.loads(line))

%%writefile tools/make_pairs_nb.py
import json, random, argparse
from pathlib import Path

def norm_box(box, W, H):
    x1,y1,x2,y2 = box
    def nv(v,m):
        v = max(0, min(m-1, v))
        return int(1000 * (v/float(m)))
    return [nv(x1,W), nv(y1,H), nv(x2,W), nv(y2,H)]

def repeat_box_for_tokens(text, box, max_tokens=256):
    n = min(len((text or "").split()), max_tokens)
    return [box] * max(1, n)

def make_pairs_nb(input_path: str, out_dir: str, valid_ratio: float=0.1, seed: int=13):
    rng = random.Random(seed)
    in_path = Path(input_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            if not page.get("blocks") or not page.get("reading_order") or len(page["reading_order"]) < 2:
                continue
            if not page.get("page_size") or len(page["page_size"]) != 2:
                continue
            W, H = page["page_size"]
            if not W or not H or W <= 0 or H <= 0:
                continue

            ro = page["reading_order"]
            id2blk = {b["id"]: b for b in page["blocks"] if "id" in b}

            # ---- 양성쌍: (i, i+1)
            for i in range(len(ro)-1):
                a = id2blk.get(ro[i]); b = id2blk.get(ro[i+1])
                if a is None or b is None: continue
                boxA = norm_box(a["box"], W, H); boxB = norm_box(b["box"], W, H)
                wbA = [norm_box(bb, W, H) for bb in a.get("word_boxes", [])] or repeat_box_for_tokens(a.get("text",""), boxA)
                wbB = [norm_box(bb, W, H) for bb in b.get("word_boxes", [])] or repeat_box_for_tokens(b.get("text",""), boxB)
                samples.append({
                    "image_path": page.get("page_image",""),
                    "text_a": a.get("text",""),
                    "boxes_a": wbA,
                    "text_b": b.get("text",""),
                    "boxes_b": wbB,
                    "label": 1
                })

            # ---- 음성쌍: 근접(하드) + 랜덤(이지) 혼합 → 2개 생성(하드1+랜덤1)
            seen_neg = set()  # (a_id, b_id) 중복 방지
            for i in range(len(ro)-2):
                a_id = ro[i]
                left  = ro[i-1] if i>0 else None
                right = ro[i+1]

                # 후보: 좌우 이웃, 자기 자신 제외
                cand_all = [x for x in ro if x not in (left, a_id, right)]
                if not cand_all:
                    continue

                a = id2blk.get(a_id)
                if a is None:
                    continue

                # --- 하드 후보: 같은 컬럼/유사 y-band
                ax1, ay1, ax2, ay2 = a["box"]
                aw = ax2-ax1; ah = ay2-ay1
                axc = (ax1+ax2)/2.0

                hard_cands = []
                for j in cand_all:
                    b = id2blk.get(j)
                    if b is None:
                        continue
                    bx1, by1, bx2, by2 = b["box"]
                    bw = bx2-bx1; bh = by2-by1
                    bxc = (bx1+bx2)/2.0

                    same_col = abs(axc - bxc) <= 0.6 * max(aw, bw)
                    close_y  = abs(by1 - ay1) <= int(0.6 * max(ah, bh))
                    if same_col or close_y:
                        hard_cands.append(j)

                # pick1: 하드에서 하나, 없으면 랜덤에서
                pick_hard = rng.choice(hard_cands) if hard_cands else rng.choice(cand_all)
                # pick2: 랜덤 하나(하드와 다르게)
                rest = [c for c in cand_all if c != pick_hard]
                if not rest:
                    rest = cand_all
                pick_rand = rng.choice(rest)

                for pick in (pick_hard, pick_rand):
                    if (a_id, pick) in seen_neg:
                        continue
                    seen_neg.add((a_id, pick))

                    b = id2blk.get(pick)
                    if b is None:
                        continue
                    boxA = norm_box(a["box"], W, H); boxB = norm_box(b["box"], W, H)
                    wbA = [norm_box(bb, W, H) for bb in a.get("word_boxes", [])] or repeat_box_for_tokens(a.get("text",""), boxA)
                    wbB = [norm_box(bb, W, H) for bb in b.get("word_boxes", [])] or repeat_box_for_tokens(b.get("text",""), boxB)
                    samples.append({
                        "image_path": page.get("page_image",""),
                        "text_a": a.get("text",""),
                        "boxes_a": wbA,
                        "text_b": b.get("text",""),
                        "boxes_b": wbB,
                        "label": 0
                    })

    rng.shuffle(samples)
    n = len(samples)
    n_valid = max(1, int(n*valid_ratio))
    train, valid = samples[n_valid:], samples[:n_valid]

    with open(out_dir/"train.jsonl", "w", encoding="utf-8") as f:
        for s in train: f.write(json.dumps(s, ensure_ascii=False)+"\n")
    with open(out_dir/"valid.jsonl", "w", encoding="utf-8") as f:
        for s in valid: f.write(json.dumps(s, ensure_ascii=False)+"\n")
    print(f"[make_pairs_nb] train={len(train)} valid={len(valid)} saved to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="data/pages.jsonl")
    ap.add_argument("--outdir", default="data/pairs")
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    make_pairs_nb(args.pages, args.outdir, args.valid_ratio, args.seed)

# pages.jsonl -> next-pair 학습용 데이터(train/valid) 생성
import json, random
from pathlib import Path

def make_pairs_nb(input_path: str, out_dir: str):
    def norm_box(box, W, H):
        x1,y1,x2,y2 = box
        clamp = lambda v, m: int(1000*max(0, min(1, v/float(m))))
        return [clamp(x1,W), clamp(y1,H), clamp(x2,W), clamp(y2,H)]

    def repeat_box_for_tokens(text, box, max_tokens=256):
        n = min(len((text or "").split()), max_tokens)
        return [box]*max(1, n)  # 텍스트 없어도 최소 1개 보장

    rng = random.Random(13)
    in_path = Path(input_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    assert in_path.exists() and in_path.stat().st_size > 0, f"missing {in_path}"

    samples = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            if not page.get("blocks") or not page.get("reading_order") or len(page["reading_order"]) < 2:
                continue
            if not page.get("page_size") or len(page["page_size"]) != 2:
                continue
            W, H = page["page_size"]
            if not W or not H or W <= 0 or H <= 0:
                continue

            ro = page["reading_order"]
            id2blk = {b["id"]: b for b in page["blocks"] if "id" in b}

            # 양성쌍
            for i in range(len(ro)-1):
                a = id2blk.get(ro[i]); b = id2blk.get(ro[i+1])
                if a is None or b is None: continue
                boxA = norm_box(a["box"], W, H); boxB = norm_box(b["box"], W, H)
                wbA = [norm_box(bb, W, H) for bb in a.get("word_boxes", [])] or repeat_box_for_tokens(a.get("text",""), boxA)
                wbB = [norm_box(bb, W, H) for bb in b.get("word_boxes", [])] or repeat_box_for_tokens(b.get("text",""), boxB)
                samples.append({
                    "image_path": page.get("page_image",""),
                    "text_a": a.get("text",""),
                    "boxes_a": wbA,
                    "text_b": b.get("text",""),
                    "boxes_b": wbB,
                    "label": 1
                })

            # 음성쌍 2개(하드1+랜덤1)
            seen_neg = set()
            for i in range(len(ro)-2):
                a_id = ro[i]
                left  = ro[i-1] if i>0 else None
                right = ro[i+1]
                cand_all = [x for x in ro if x not in (left, a_id, right)]
                if not cand_all:
                    continue
                a = id2blk.get(a_id)
                if a is None:
                    continue

                ax1, ay1, ax2, ay2 = a["box"]
                aw = ax2-ax1; ah = ay2-ay1
                axc = (ax1+ax2)/2.0

                hard_cands = []
                for j in cand_all:
                    b = id2blk.get(j)
                    if b is None:
                        continue
                    bx1, by1, bx2, by2 = b["box"]
                    bw = bx2-bx1; bh = by2-by1
                    bxc = (bx1+bx2)/2.0
                    same_col = abs(axc - bxc) <= 0.6 * max(aw, bw)
                    close_y  = abs(by1 - ay1) <= int(0.6 * max(ah, bh))
                    if same_col or close_y:
                        hard_cands.append(j)

                pick_hard = rng.choice(hard_cands) if hard_cands else rng.choice(cand_all)
                rest = [c for c in cand_all if c != pick_hard] or cand_all
                pick_rand = rng.choice(rest)

                for pick in (pick_hard, pick_rand):
                    if (a_id, pick) in seen_neg:
                        continue
                    seen_neg.add((a_id, pick))
                    b = id2blk.get(pick)
                    if b is None:
                        continue
                    boxA = norm_box(a["box"], W, H); boxB = norm_box(b["box"], W, H)
                    wbA = [norm_box(bb, W, H) for bb in a.get("word_boxes", [])] or repeat_box_for_tokens(a.get("text",""), boxA)
                    wbB = [norm_box(bb, W, H) for bb in b.get("word_boxes", [])] or repeat_box_for_tokens(b.get("text",""), boxB)
                    samples.append({
                        "image_path": page.get("page_image",""),
                        "text_a": a.get("text",""),
                        "boxes_a": wbA,
                        "text_b": b.get("text",""),
                        "boxes_b": wbB,
                        "label": 0
                    })

    rng.shuffle(samples)
    n = len(samples); n_valid = max(1, int(n*0.1))
    train, valid = samples[n_valid:], samples[:n_valid]
    with open(out_dir/"train.jsonl", "w", encoding="utf-8") as f:
        for s in train: f.write(json.dumps(s, ensure_ascii=False)+"\n")
    with open(out_dir/"valid.jsonl", "w", encoding="utf-8") as f:
        for s in valid: f.write(json.dumps(s, ensure_ascii=False)+"\n")
    print(f"[make_pairs_nb] train={len(train)} valid={len(valid)} saved to {out_dir}")

# 실행
from pathlib import Path
Path("data/pairs").mkdir(parents=True, exist_ok=True)
make_pairs_nb("data/pages.jsonl", "data/pairs")

# 확인
for p in ["data/pairs/train.jsonl","data/pairs/valid.jsonl"]:
    q=Path(p)
    print(p, "OK", q.stat().st_size, "bytes" if q.exists() else "MISSING")

!pip -q install -U "transformers==4.44.2" "peft==0.12.0" "accelerate>=0.33.0" "datasets>=2.20.0"

!python training/train_layoutlm_nextpair_lora.py \
  --train data/pairs/train.jsonl \
  --valid data/pairs/valid.jsonl \
  --out runs/nextpair_lora \
  --base-model microsoft/layoutlm-base-uncased \
  --epochs 8 --bs 8 --ga 2 \
  --lr 1e-4 --warmup-ratio 0.1 --weight-decay 0.01 --max-grad-norm 1.0 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05

# ==== 설정 ====
BASE_MODEL  = "microsoft/layoutlm-base-uncased"
ADAPTER_DIR = "/content/runs/nextpair_lora"   # <- 네가 방금 학습 저장한 폴더
IN_PAGES    = "data/pages.jsonl"
OUT_PAGES   = "data/pages_fused.jsonl"
LAMBDA_GEO  = 0.5   # geo:LM 가중치 (0.5부터 시작, 0.3~0.7 탐색 권장)
N_PASSES    = 2     # 로컬 스왑 패스 횟수(작게 1~3 권장)

# ==== 코드 ====
import json, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, LayoutLMForSequenceClassification
from peft import PeftModel

def _expand_or_trim_boxes(boxes, n_tokens):
    if not boxes: return [[0,0,0,0]]*n_tokens
    out=[]; i=0
    while len(out)<n_tokens:
        out.append(boxes[min(i, len(boxes)-1)])
        i+=1
    return out[:n_tokens]

def encode_pair(tok, text_a, boxes_a, text_b, boxes_b, max_len=512):
    enc_a = tok(text_a or "", add_special_tokens=False)
    enc_b = tok(text_b or "", add_special_tokens=False)
    room = max_len - 3
    la = min(len(enc_a["input_ids"]), room//2)
    lb = min(len(enc_b["input_ids"]), room - la)

    ids  = [tok.cls_token_id] + enc_a["input_ids"][:la] + [tok.sep_token_id] + enc_b["input_ids"][:lb] + [tok.sep_token_id]
    attn = [1]*len(ids)
    ttid = [0]*(1+la+1) + [1]*(lb+1)

    ba = _expand_or_trim_boxes(boxes_a, la)
    bb = _expand_or_trim_boxes(boxes_b, lb)
    bbox = [[0,0,0,0]] + ba + [[0,0,0,0]] + bb + [[0,0,0,0]]

    return {
        "input_ids":     torch.tensor([ids], dtype=torch.long),
        "attention_mask":torch.tensor([attn], dtype=torch.long),
        "token_type_ids":torch.tensor([ttid], dtype=torch.long),
        "bbox":          torch.tensor([bbox], dtype=torch.long),
    }

# 모델 로드 (LoRA 어댑터 결합)
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
base = LayoutLMForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device).eval()

@torch.no_grad()
def prob_adjacent(a, b, id2blk):
    A = id2blk[a]; B = id2blk[b]
    enc = encode_pair(
        tok,
        A.get("text",""), A.get("word_boxes", []),
        B.get("text",""), B.get("word_boxes", []),
        max_len=512
    )
    enc = {k:v.to(device) for k,v in enc.items()}
    logits = model(**enc).logits
    prob = torch.softmax(logits, dim=-1)[0,1].item()  # label=1 (adjacent)
    return prob

def fuse_order_for_page(page, lam=LAMBDA_GEO, passes=N_PASSES):
    blocks = page.get("blocks", [])
    order  = list(page.get("reading_order", []))
    if len(order) < 3:   # 스왑해봤자 의미 적음
        return order

    id2blk = {b["id"]: b for b in blocks}
    # 여러 번 패스 돌며 로컬 (j,k) 스왑이 더 좋으면 교환 (bubble-like)
    for _ in range(passes):
        changed = False
        for idx in range(len(order)-2):
            i, j, k = order[idx], order[idx+1], order[idx+2]

            # 현재 구성 점수: i->j, j->k (둘 다 '현재 인접' 이므로 geo=1)
            s_cur = lam*1.0 + (1-lam)*prob_adjacent(i, j, id2blk)
            s_cur += lam*1.0 + (1-lam)*prob_adjacent(j, k, id2blk)

            # 스왑 구성 점수: i->k, k->j (스왑하면 이 두 간선이 인접이 됨 → geo=1)
            s_swap = lam*1.0 + (1-lam)*prob_adjacent(i, k, id2blk)
            s_swap += lam*1.0 + (1-lam)*prob_adjacent(k, j, id2blk)

            if s_swap > s_cur:
                order[idx+1], order[idx+2] = k, j
                changed = True
        if not changed:
            break
    return order

# 실행: pages.jsonl → pages_fused.jsonl
in_path  = Path(IN_PAGES)
out_path = Path(OUT_PAGES)
assert in_path.exists(), f"missing: {IN_PAGES}"

n_in=0; n_out=0
with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        n_in += 1
        page = json.loads(line)
        try:
            fused_order = fuse_order_for_page(page, LAMBDA_GEO, N_PASSES)
            page["reading_order_fused"] = fused_order
        except Exception as e:
            # 실패 시 원래 순서 그대로 백업
            page["reading_order_fused"] = page.get("reading_order", [])
        fout.write(json.dumps(page, ensure_ascii=False) + "\n")
        n_out += 1

print(f"[OK] fused pages written: {n_out} -> {OUT_PAGES}")

# 간단 미리보기 (앞 2페이지)
with open(OUT_PAGES, "r", encoding="utf-8") as f:
    import itertools, pprint
    for page in itertools.islice(f, 2):
        page = json.loads(page)
        print("image:", page.get("page_image",""))
        print("geo:",   page.get("reading_order", [])[:10])
        print("fused:", page.get("reading_order_fused", [])[:10])
        print("-"*60)
