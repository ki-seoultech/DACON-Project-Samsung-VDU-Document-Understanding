%%writefile prep_data.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
import sys
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

# ------------------------------
# 전역 설정
# ------------------------------
CLASSES = ["title", "subtitle", "text", "image", "table", "equation"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

# 데이터셋별 카테고리 매핑표 (source_name -> target_name or None)
# 범용적으로 최대치 포용 + 안전제거(헤더/푸터/페이지번호/참고문헌 등)
MAPPINGS: Dict[str, Dict[str, Optional[str]]] = {
    # DocLayNet (11 classes; 일부 명칭은 논문/가이드 기준, 대소문자/하이픈 변형 허용)
    "doclaynet": {
        "title": "title",
        "section-header": "subtitle",  # 부제/섹션헤더를 subtitle로 캐스팅
        "section_header": "subtitle",
        "text": "text",
        "caption": "text",
        "list-item": "text",
        "list": "text",
        "figure": "image",
        "picture": "image",
        "table": "table",
        "formula": "equation",  # 문헌에 formula 라벨 존재 보고됨
        "equation": "equation",
        # 제거 대상
        "footnote": None,
        "page-footer": None,
        "page_header": None,
        "page-header": None,
        "page-footer": None,
        "page number": None,
        "page-number": None,
        "reference": None,
        "bibliography": None,
    },
    # PubLayNet (text, title, list, table, figure)
    "publaynet": {
        "title": "title",
        "text": "text",
        "list": "text",
        "table": "table",
        "figure": "image",
    },
    # DocBank (다양; Equation/Title/Section/Paragraph/Figure/Table/Caption/List 등)
    "docbank": {
        "title": "title",
        "section": "subtitle",      # (Sub)section류를 subtitle로 취급
        "section*": "subtitle",
        "subsection": "subtitle",
        "subsection*": "subtitle",
        "paragraph": "text",
        "abstract": "text",
        "caption": "text",
        "list": "text",
        "equation": "equation",
        "formula": "equation",
        "figure": "image",
        "table": "table",
        # 제거 후보
        "footnote": None,
        "header": None,
        "footer": None,
        "page-header": None,
        "page-footer": None,
        "reference": None,
        "bibliography": None,
    },
    # PubTables-1M (VOC: class names: table, table rotated)
    "pubtables1m": {
        "table": "table",
        "table rotated": "table",
    },
}

# ------------------------------
# 유틸
# ------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def coco_load(ann_path: Path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = {c["id"]: c.get("name", str(c["id"])).lower() for c in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)
    return images, anns_by_img, cats


def to_yolo_bbox(x, y, w, h, iw, ih):
    # COCO x,y,w,h -> YOLO x_c, y_c, w, h (normalized)
    x_c = (x + w / 2.0) / iw
    y_c = (y + h / 2.0) / ih
    return x_c, y_c, w / iw, h / ih


def write_yolo_label(lines: List[str], out_path: Path):
    if not lines:
        return
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------
# COCO -> YOLO
# ------------------------------

def cmd_map_coco(args):
    ds = args.dataset.lower()
    mapping = MAPPINGS.get(ds)
    if mapping is None:
        raise ValueError(f"Unknown dataset mapping: {ds}")

    ann_path = Path(args.ann)
    images_dir = Path(args.images_dir)
    out_root = Path(args.out_root)

    out_images = out_root / "images" / args.split
    out_labels = out_root / "labels" / args.split
    ensure_dir(out_images)
    ensure_dir(out_labels)
    ensure_dir(out_root / "meta")

    # classes 파일 기록
    classes_yaml = out_root / "meta" / "classes.yaml"
    if not classes_yaml.exists():
        with open(classes_yaml, "w", encoding="utf-8") as f:
            f.write("names: [title, subtitle, text, image, table, equation]\n")

    images, anns_by_img, cats = coco_load(ann_path)

    kept, skipped = 0, 0
    for img_id, img in images.items():
        file_name = img.get("file_name")
        iw, ih = img.get("width"), img.get("height")
        if not file_name or iw is None or ih is None:
            skipped += 1
            continue
        src_img = images_dir / file_name
        if not src_img.exists():
            # 일부 데이터셋은 하위 폴더가 있을 수 있음 → 전체 트리에서 검색 (비용↑)
            # 여기서는 단순히 스킵
            skipped += 1
            continue

        yolo_lines = []
        for ann in anns_by_img.get(img_id, []):
            cat_name = cats.get(ann["category_id"], str(ann["category_id"]).lower())
            cat_name = cat_name.lower()
            # mapping 키 보정(하이픈/언더스코어/공백 제거)
            k = cat_name.replace(" ", "-").replace("_", "-")
            tgt = mapping.get(k, mapping.get(cat_name))
            if tgt is None:
                continue
            if tgt not in CLASS2ID:
                continue

            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 1 or h <= 1:
                continue
            x_c, y_c, ww, hh = to_yolo_bbox(x, y, w, h, iw, ih)
            cid = CLASS2ID[tgt]
            yolo_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}")

        if not yolo_lines:
            # 해당 이미지에서 우리 타깃 클래스가 없으면 스킵
            continue

        # 출력 파일명: datasetprefix_filename
        out_name = f"{ds}__{Path(file_name).stem}{Path(file_name).suffix}"
        dst_img = out_images / out_name
        symlink_or_copy(src_img, dst_img)
        write_yolo_label(yolo_lines, out_labels / f"{Path(out_name).stem}.txt")
        kept += 1

    print(f"[map-coco:{ds}] kept={kept}, skipped={skipped}, out_root={out_root}")


# ------------------------------
# VOC(XML) -> YOLO (PubTables-1M)
# ------------------------------

def parse_voc_xml(xml_path: Path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    iw = int(size.findtext("width"))
    ih = int(size.findtext("height"))
    objs = []  # (name, xmin, ymin, xmax, ymax)
    for obj in root.findall("object"):
        name = obj.findtext("name").lower()
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        objs.append((name, xmin, ymin, xmax, ymax))
    return iw, ih, objs


def cmd_map_voc(args):
    ds = args.dataset.lower()
    mapping = MAPPINGS.get(ds)
    if mapping is None:
        raise ValueError(f"Unknown dataset mapping: {ds}")

    voc_dir = Path(args.voc_dir)
    images_dir = Path(args.images_dir)
    out_root = Path(args.out_root)

    out_images = out_root / "images" / args.split
    out_labels = out_root / "labels" / args.split
    ensure_dir(out_images)
    ensure_dir(out_labels)
    ensure_dir(out_root / "meta")

    classes_yaml = out_root / "meta" / "classes.yaml"
    if not classes_yaml.exists():
        with open(classes_yaml, "w", encoding="utf-8") as f:
            f.write("names: [title, subtitle, text, image, table, equation]\n")

    xmls = list(voc_dir.rglob("*.xml"))
    kept = 0
    for x in xmls:
        iw, ih, objs = parse_voc_xml(x)
        yolo_lines = []
        for name, xmin, ymin, xmax, ymax in objs:
            tgt = mapping.get(name)
            if tgt != "table":
                continue
            w = xmax - xmin
            h = ymax - ymin
            if w <= 1 or h <= 1:
                continue
            x_c = (xmin + w / 2.0) / iw
            y_c = (ymin + h / 2.0) / ih
            ww = w / iw
            hh = h / ih
            cid = CLASS2ID["table"]
            yolo_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}")
        if not yolo_lines:
            continue

        img_name = x.with_suffix(".jpg").name
        # 일부는 .png 가능 → 우선 jpg, 없으면 png 시도
        src_img = images_dir / img_name
        if not src_img.exists():
            alt = images_dir / x.with_suffix(".png").name
            if alt.exists():
                src_img = alt
            else:
                continue

        out_name = f"{ds}__{src_img.name}"
        symlink_or_copy(src_img, out_images / out_name)
        write_yolo_label(yolo_lines, out_labels / f"{Path(out_name).stem}.txt")
        kept += 1

    print(f"[map-voc:{ds}] kept={kept}, out_root={out_root}")


# ------------------------------
# 클래스 균형 샘플링 (이미지 단위)
# ------------------------------

def cmd_sample_balanced(args):
    yolo_root = Path(args.yolo_root)
    out_root = Path(args.out_split_root)
    ensure_dir(out_root / "images/train")
    ensure_dir(out_root / "labels/train")

    target = int(args.target_per_class)

    # 이미지별 포함 클래스 집계
    label_files = list((yolo_root / "labels" / "train").glob("*.txt"))
    img_for_lbl = defaultdict(list)  # class_id -> [image_path]
    img_path_by_label = {}

    for lf in label_files:
        img = (yolo_root / "images" / "train" / (lf.stem + ".jpg"))
        if not img.exists():
            img = (yolo_root / "images" / "train" / (lf.stem + ".png"))
        if not img.exists():
            continue
        with open(lf, "r", encoding="utf-8") as f:
            classes = set()
            for line in f:
                try:
                    cid = int(line.strip().split()[0])
                    classes.add(cid)
                except Exception:
                    pass
        for cid in classes:
            img_for_lbl[cid].append(img)
        img_path_by_label[lf] = img

    # 균형 샘플링: 각 클래스별 target만큼 이미지 선택(중복 가능, set으로 고정)
    selected = set()
    rng = random.Random(args.seed)
    for cid in range(len(CLASSES)):
        pool = img_for_lbl.get(cid, [])
        rng.shuffle(pool)
        for p in pool[:target]:
            selected.add(p)

    # 선택본 복사/링크
    for img in selected:
        lbl = (yolo_root / "labels" / "train" / (img.stem + ".txt"))
        if not lbl.exists():
            continue
        symlink_or_copy(img, out_root / "images/train" / img.name)
        symlink_or_copy(lbl, out_root / "labels/train" / lbl.name)

    # meta/classes.yaml 동기화
    meta_src = yolo_root / "meta" / "classes.yaml"
    if meta_src.exists():
        ensure_dir(out_root / "meta")
        symlink_or_copy(meta_src, out_root / "meta" / "classes.yaml")

    print(f"[sample-balanced] selected_images={len(selected)} out_root={out_root}")


# ------------------------------
# Split train/val (이미지 단위 층화 근사)
# ------------------------------

def cmd_split(args):
    yolo_root = Path(args.yolo_root)
    val_ratio = float(args.val_ratio)
    seed = int(args.seed)

    images_dir = yolo_root / "images/train"
    labels_dir = yolo_root / "labels/train"

    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    rng = random.Random(seed)

    # 간단 층화: 각 이미지의 대표 클래스(최빈) 기준 버킷화
    buckets = defaultdict(list)
    for img in imgs:
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        cnt = Counter()
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cnt[int(parts[0])] += 1
        rep = cnt.most_common(1)[0][0] if cnt else -1
        buckets[rep].append(img)

    val_set = set()
    for rep, lst in buckets.items():
        k = max(1, int(round(len(lst) * val_ratio)))
        rng.shuffle(lst)
        for p in lst[:k]:
            val_set.add(p)

    # move/symlink
    for phase in ["train", "val"]:
        ensure_dir(yolo_root / f"images/{phase}")
        ensure_dir(yolo_root / f"labels/{phase}")

    # 먼저 전체를 train으로 복사해두고, val은 별도로 링크
    # (여기서는 간단히: 선택된 val만 새 폴더로 링크, train은 기존 유지)
    for img in val_set:
        lbl = labels_dir / (img.stem + ".txt")
        symlink_or_copy(img, yolo_root / f"images/val/{img.name}")
        if lbl.exists():
            symlink_or_copy(lbl, yolo_root / f"labels/val/{lbl.name}")

    print(f"[split] total={len(imgs)} val={len(val_set)} val_ratio={val_ratio}")


# ------------------------------
# Reading Order Pair Generation (약지도)
# ------------------------------

def guess_reading_order(boxes: List[Tuple[float, float, float, float]], iw: int, ih: int) -> List[int]:
    """간단한 좌->우, 상->하 정렬 규칙으로 순서를 근사.
    boxes: (x_center, y_center, w, h) (YOLO norm)
    return: 인덱스 순열
    """
    # 절대 좌표로 변환
    abs_boxes = []
    for i, (xc, yc, w, h) in enumerate(boxes):
        x = (xc - w / 2.0) * iw
        y = (yc - h / 2.0) * ih
        abs_boxes.append((i, x, y))
    # y 우선, x 보조 정렬
    order = sorted(range(len(abs_boxes)), key=lambda k: (abs_boxes[k][2] // 32, abs_boxes[k][1]))
    return order


def tta_boxes(boxes: List[Tuple[float, float, float, float]], aug: str) -> List[Tuple[float, float, float, float]]:
    if aug == "flip":
        return [(1 - xc, yc, w, h) for (xc, yc, w, h) in boxes]
    if aug == "rotate":
        # 90도 회전 (시계) 기준 단순 변환
        return [(yc, 1 - xc, h, w) for (xc, yc, w, h) in boxes]
    return boxes


def cmd_gen_order_pairs(args):
    yolo_root = Path(args.yolo_root)
    out_csv = Path(args.out_csv)
    tta_list = []
    if args.tta:
        tta_list = [t.strip() for t in args.tta.split(',') if t.strip()]

    images_dir = yolo_root / "images/train"
    labels_dir = yolo_root / "labels/train"
    ensure_dir(out_csv.parent)

    rows = []
    pair_id = 0
    for img in images_dir.iterdir():
        if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        # 이미지 크기
        iw = ih = None
        if Image is not None:
            try:
                with Image.open(img) as im:
                    iw, ih = im.size
            except Exception:
                pass
        if iw is None or ih is None:
            # YOLO norm만으로도 상대 순서 근사 가능하므로 임의값
            iw, ih = 1000, 1000

        # 라벨 로드
        boxes = []
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if cid not in (CLASS2ID["title"], CLASS2ID["subtitle"], CLASS2ID["text"]):
                    # 읽기순서 후보는 텍스트성 블록 위주로
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((xc, yc, w, h))

        if len(boxes) < 2:
            continue

        # 기본 + TTA 변형으로 순서 근사
        variants = [("orig", boxes)]
        for t in tta_list:
            variants.append((t, tta_boxes(boxes, t)))

        for tag, bx in variants:
            order = guess_reading_order(bx, iw, ih)
            # 인접 양성 페어 생성
            for i in range(len(order) - 1):
                src = order[i]
                dst = order[i + 1]
                rows.append([str(img), f"{pair_id}", src, dst, 1])
                pair_id += 1
            # 음성 페어(하드네거티브): 비연속 임의 쌍
            idxs = list(range(len(bx)))
            random.shuffle(idxs)
            for i in range(0, len(idxs) - 1, 2):
                if abs(i - (i + 1)) <= 1:
                    continue
                rows.append([str(img), f"{pair_id}", idxs[i], idxs[i + 1], 0])
                pair_id += 1

    # 저장
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("image_path,pair_id,src_idx,dst_idx,label\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"[gen-order-pairs] pairs={len(rows)} out_csv={out_csv}")


# ------------------------------
# Main
# ------------------------------

def main():
    p = argparse.ArgumentParser(description="VDU Data Prep Pipeline")
    sub = p.add_subparsers(dest="cmd")

    # map-coco
    s = sub.add_parser("map-coco")
    s.add_argument("--dataset", required=True, choices=["doclaynet", "publaynet", "docbank"])  # COCO류
    s.add_argument("--ann", required=True, help="COCO annotation json path")
    s.add_argument("--images-dir", required=True, help="images root")
    s.add_argument("--out-root", required=True)
    s.add_argument("--split", default="train")
    s.set_defaults(func=cmd_map_coco)

    # map-voc
    s = sub.add_parser("map-voc")
    s.add_argument("--dataset", required=True, choices=["pubtables1m"])  # VOC류
    s.add_argument("--voc-dir", required=True, help="VOC xml dir")
    s.add_argument("--images-dir", required=True, help="images root")
    s.add_argument("--out-root", required=True)
    s.add_argument("--split", default="train")
    s.set_defaults(func=cmd_map_voc)

    # sample-balanced
    s = sub.add_parser("sample-balanced")
    s.add_argument("--yolo-root", required=True)
    s.add_argument("--target-per-class", required=True, type=int)
    s.add_argument("--out-split-root", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_sample_balanced)

    # split
    s = sub.add_parser("split")
    s.add_argument("--yolo-root", required=True)
    s.add_argument("--val-ratio", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_split)

    # reading order pairs
    s = sub.add_parser("gen-order-pairs")
    s.add_argument("--yolo-root", required=True)
    s.add_argument("--out-csv", required=True)
    s.add_argument("--tta", default="", help="comma-separated: rotate,flip")
    s.set_defaults(func=cmd_gen_order_pairs)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

import os, glob

# 후보 경로 (필요 시 여기에 더 추가해도 OK)
coco_candidates = [
    "/content/datasets/doclaynet/COCO/train.json",
    "/content/DocLayNet_core/COCO/train.json",
    "/content/DocLayNet_core.zip/COCO/train.json",
    "/content/COCO/train.json",
]
png_dir_candidates = [
    "/content/datasets/doclaynet/PNG",
    "/content/DocLayNet_core/PNG",
    "/content/DocLayNet_core.zip/PNG",
    "/content/PNG",
]

ANN_PATH = next((p for p in coco_candidates if os.path.isfile(p)), None)
if ANN_PATH is None:
    # 마지막 수단: 전체 탐색(느릴 수 있음)
    hits = glob.glob("/content/**/COCO/train.json", recursive=True)
    ANN_PATH = hits[0] if hits else None

PNG_DIR = next((d for d in png_dir_candidates if os.path.isdir(d)), None)
if PNG_DIR is None:
    hits = glob.glob("/content/**/PNG", recursive=True)
    PNG_DIR = hits[0] if hits else None

print("ANN_PATH =", ANN_PATH)
print("PNG_DIR  =", PNG_DIR)

assert ANN_PATH and PNG_DIR, "COCO/train.json 또는 PNG 폴더를 찾지 못했습니다."

# ✅ PNG 링크 교정 + 전체 파이프라인 재실행 (kept=0 문제 해결)
import os, glob, json, subprocess, sys, random
from pathlib import Path
import pandas as pd

import yaml
from pathlib import Path


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("[cmd]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout)
    if p.returncode != 0:
        print("=== STDERR ===")
        print(p.stderr)
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)

# 0) COCO train.json
coco_train = next((p for p in glob.glob("/content/**/COCO/train.json", recursive=True) if os.path.isfile(p)), None)
assert coco_train, "COCO/train.json을 찾지 못했습니다."
with open(coco_train, "r", encoding="utf-8") as f:
    coco = json.load(f)
names = [im.get("file_name") for im in coco.get("images", []) if im.get("file_name")]
sample = names[:80] if len(names) >= 80 else names

def score_png_dir(png_dir: str) -> int:
    """COCO file_name을 실제로 찾는 hit 수(정확 join + basename fallback)"""
    hit = 0
    for fn in sample:
        p1 = os.path.join(png_dir, fn)
        p2 = os.path.join(png_dir, os.path.basename(fn))
        if os.path.exists(p1) or os.path.exists(p2):
            hit += 1
    return hit

# 1) PNG 후보 수집 (realpath로 평가)
candidates = []
manual = ["/content/PNG", "/content/DocLayNet_core/PNG", "/content/datasets/doclaynet/PNG"]
for p in manual:
    if os.path.isdir(p): candidates.append(p)
for p in glob.glob("/content/**/PNG", recursive=True):
    if os.path.isdir(p) and p not in candidates:
        candidates.append(p)

scored = []
for d in candidates:
    real = os.path.realpath(d)
    if not os.path.isdir(real):  # 깨진 링크 제외
        continue
    s = score_png_dir(real)
    scored.append((d, real, s))
scored.sort(key=lambda x: x[2], reverse=True)

assert scored and scored[0][2] > 3, f"COCO 파일명과 일치하는 PNG 디렉터리를 찾지 못했습니다. 후보(상위3): {scored[:3]}"
best_path, best_real, best_hit = scored[0]
print("COCO train.json =", coco_train)
print(f"✅ 선택된 PNG 디렉터리 = {best_real} (hits={best_hit})")

# 2) 깨진/순환 링크 교정: /content/datasets/doclaynet/PNG → best_real
link_path = "/content/datasets/doclaynet/PNG"
Path("/content/datasets/doclaynet").mkdir(parents=True, exist_ok=True)
# 이미 존재하면 제거(디렉터리/링크 모두 커버)
if os.path.islink(link_path) or os.path.exists(link_path):
    try: os.remove(link_path)
    except IsADirectoryError:
        import shutil; shutil.rmtree(link_path)
os.symlink(best_real, link_path)
print("🔗 /content/datasets/doclaynet/PNG →", os.path.realpath(link_path))

# 3) 출력 루트
YOLO_UNI = "/content/workspace/unified_yolo"
YOLO_BAL = "/content/workspace/dataset_balanced"
PAIR_CSV = f"{YOLO_BAL}/meta/order_pairs_train.csv"
os.makedirs(YOLO_UNI, exist_ok=True)
os.makedirs(YOLO_BAL, exist_ok=True)
os.makedirs(os.path.dirname(PAIR_CSV), exist_ok=True)

# (선택) 기존 변환물 정리: kept=0 상태라면 labels/train 비어있음 → 안전히 덮어쓰기
# 필요 시 아래 주석 제거하여 깨끗이 재생성
# import shutil
# shutil.rmtree(YOLO_UNI, ignore_errors=True); os.makedirs(YOLO_UNI, exist_ok=True)
# shutil.rmtree(YOLO_BAL, ignore_errors=True); os.makedirs(YOLO_BAL, exist_ok=True); os.makedirs(os.path.dirname(PAIR_CSV), exist_ok=True)

# 4) COCO → YOLO (train/val)
run([sys.executable, "prep_data.py", "map-coco",
     "--dataset", "doclaynet",
     "--ann", coco_train,
     "--images-dir", os.path.realpath(link_path),
     "--out-root", YOLO_UNI,
     "--split", "train"])

coco_val = next((p for p in glob.glob("/content/**/COCO/val.json", recursive=True) if os.path.isfile(p)), None)
if coco_val:
    run([sys.executable, "prep_data.py", "map-coco",
         "--dataset", "doclaynet",
         "--ann", coco_val,
         "--images-dir", os.path.realpath(link_path),
         "--out-root", YOLO_UNI,
         "--split", "val"])
else:
    print("[warn] val.json 없음 — split 단계에서 val 구성")

# 5) 클래스 균형 샘플링 + split
run([sys.executable, "prep_data.py", "sample-balanced",
     "--yolo-root", YOLO_UNI,
     "--target-per-class", "12000",
     "--out-split-root", YOLO_BAL,
     "--seed", "42"])

run([sys.executable, "prep_data.py", "split",
     "--yolo-root", YOLO_BAL,
     "--val-ratio", "0.15",
     "--seed", "42"])

# 6) Reading Order 페어 생성
run([sys.executable, "prep_data.py", "gen-order-pairs",
     "--yolo-root", YOLO_BAL,
     "--out-csv", PAIR_CSV,
     "--tta", "rotate,flip"])

# 7) pos:neg=1:1 보정 (역방향 + 랜덤 비연속)
TEXT_CLASS = {0,1,2}
def count_text_boxes(stem, root):
    for split in ["train","val"]:
        lp = Path(root)/"labels"/split/f"{stem}.txt"
        if lp.exists():
            n = 0
            with open(lp,"r",encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip(): continue
                    c = int(float(ln.split()[0]))
                    if c in TEXT_CLASS: n += 1
            return n
    return 0

df = pd.read_csv(PAIR_CSV) if Path(PAIR_CSV).exists() else pd.DataFrame(columns=["image_path","pair_id","src_idx","dst_idx","label"])
if df.empty:
    print("[warn] pairs.csv가 비어 있습니다. 앞 단계 로그에서 kept>0인지 확인하세요.")
else:
    g = df.groupby("image_path")
    max_pair_id = 0 if df.empty else int(df["pair_id"].astype(int).max())
    rows_neg, rng = [], random.Random(42)
    for img_path, sub in g:
        sub = sub.copy()
        pos = sub[sub["label"]==1]
        need_neg = len(pos)
        if need_neg == 0:
            continue
        T = count_text_boxes(Path(img_path).stem, YOLO_BAL)
        if T < 2:
            continue
        existing = set((int(r.src_idx), int(r.dst_idx)) for r in sub.itertuples(index=False))
        candidates = []
        # 역방향
        for r in pos.itertuples(index=False):
            a,b = int(r.src_idx), int(r.dst_idx)
            if (b,a) not in existing and a!=b:
                candidates.append((b,a))
        # 랜덤 보충
        tries = 0
        while len(candidates) < need_neg and tries < need_neg*10:
            tries += 1
            a = rng.randrange(T); b = rng.randrange(T)
            if a==b: continue
            if (a,b) in existing or (a,b) in candidates: continue
            candidates.append((a,b))
        for a,b in candidates[:need_neg]:
            max_pair_id += 1
            rows_neg.append([img_path, str(max_pair_id), a, b, 0])
    if rows_neg:
        df_neg = pd.DataFrame(rows_neg, columns=["image_path","pair_id","src_idx","dst_idx","label"])
        df_out = pd.concat([df, df_neg], ignore_index=True)
        bak = Path(PAIR_CSV).with_suffix(".bak.csv")
        if not bak.exists(): df.to_csv(bak, index=False)
        df_out.to_csv(PAIR_CSV, index=False)
        print(f"[OK] neg {len(df_neg)}개 추가 → pos:neg = {int((df_out['label']==1).sum())}:{int((df_out['label']==0).sum())}")
        df = df_out
    else:
        print("[warn] neg 추가 후보 없음")

# 8) 요약
if not df.empty:
    pos = int((df["label"]==1).sum())
    neg = int((df["label"]==0).sum())
    print(f"[check] pairs rows={len(df)}  pos={pos}  neg={neg}  pos_ratio={pos/max(1,pos+neg):.2%}")


YOLO_BAL = Path("/content/workspace/dataset_balanced")
DATA_YAML = YOLO_BAL/"data.yaml"

data = {
    "path": str(YOLO_BAL),
    "train": "images/train",
    "val": "images/val",
    "names": ["title","subtitle","text","image","table","equation"]
}
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True)

print("[OK] data.yaml 생성 완료:", DATA_YAML)
print(DATA_YAML.read_text())

# VDU 데이터셋 통합 검증기 (관대 모드 + 종합 리포트)
# - data.yaml / classes.yaml 점검
# - train/val 존재/개수/미스매치
# - YOLO 라벨 형식 + 범위(±TOL) + 경계(left/right/top/bot) 체크
# - 클래스 히스토그램(샘플)
# - pairs.csv 무결성(이미지/라벨 존재, 텍스트 인덱스 범위) + pos:neg 비율
# - 요약 리포트 (soft/hard violation, 분포, 경고)

import os, glob, json, random, statistics as stats
from pathlib import Path
from collections import Counter
import pandas as pd

# ===== 경로/설정 =====
YOLO_BAL = Path("/content/workspace/dataset_balanced")
PAIR_CSV  = YOLO_BAL/"meta"/"order_pairs_train.csv"
DATA_YAML = YOLO_BAL/"data.yaml"
CLASSES_YAML = YOLO_BAL/"meta"/"classes.yaml"

CLASSES = ["title","subtitle","text","image","table","equation"]
CLASS2ID = {c:i for i,c in enumerate(CLASSES)}
TEXT_SET = {CLASS2ID["title"], CLASS2ID["subtitle"], CLASS2ID["text"]}

# 허용오차 및 동작
TOL = 2e-3        # 0.002 이내는 soft 경고
HARD_TOL = 5e-2   # 0.05 초과는 hard 경고
STRICT = False    # True면 hard violation 존재 시 AssertionError

# 라벨 샘플링 수
SAMPLE_LABELS_PER_SPLIT = 1000
RAND_SEED = 42
random.seed(RAND_SEED)

# ===== 유틸 =====
def read_first_n_lines(p, n=3):
    try:
        with open(p, "r", encoding="utf-8") as f:
            out = []
            for _ in range(n):
                out.append(next(f).rstrip("\n"))
            return out
    except Exception:
        return []

def try_load_yaml(path: Path):
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # yaml 미설치/파서실패 → 매우 단순 키 존재만 확인
        txt = path.read_text(encoding="utf-8")
        has = all(k in txt for k in ["train","val","names"])
        return {"_raw": txt, "_minimal_ok": has}

def check_files_exist(img_dir: Path, lab_dir: Path):
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    labs = sorted([p for p in lab_dir.iterdir() if p.suffix.lower()==".txt"])
    img_stems = {p.stem for p in imgs}
    lab_stems = {p.stem for p in labs}
    only_img = img_stems - lab_stems
    only_lab = lab_stems - img_stems
    return imgs, labs, only_img, only_lab

def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"label line length < 5: {line}")
    c = int(float(parts[0]))
    xc = float(parts[1]); yc = float(parts[2]); w = float(parts[3]); h = float(parts[4])
    return c, xc, yc, w, h

def within01_soft(v):   # -TOL ~ 1+TOL 허용
    return (-TOL <= v <= 1.0 + TOL)
def within01_hard(v):   # -HARD_TOL ~ 1+HARD_TOL 허용
    return (-HARD_TOL <= v <= 1.0 + HARD_TOL)

soft_violations = 0
hard_violations = 0
soft_samples = []
hard_samples = []

def validate_label_file(lab_path: Path, img_dir: Path):
    """라벨 1파일: 이미지 존재/형식/범위/경계 체크 + 클래스 카운트"""
    global soft_violations, hard_violations
    hist = Counter()

    # 이미지 존재(경고만)
    img_jpg = img_dir/(lab_path.stem + ".jpg")
    img_png = img_dir/(lab_path.stem + ".png")
    img_jpeg = img_dir/(lab_path.stem + ".jpeg")
    if not any(p.exists() for p in [img_jpg, img_png, img_jpeg]):
        print(f"[warn] image missing for {lab_path}")
        return hist

    # 라벨 라인
    with open(lab_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        print(f"[warn] empty label: {lab_path}")
        return hist

    for ln in lines:
        try:
            c, xc, yc, w, h = parse_label_line(ln)
        except Exception as e:
            print(f"[warn] parse error in {lab_path}: {e}")
            continue

        if not (0 <= c < len(CLASSES)):
            print(f"[warn] class out of range ({c}) in {lab_path}")
            continue

        # 값 범위(관대)
        values = {"xc":xc, "yc":yc, "w":w, "h":h}
        for k, v in values.items():
            if not within01_soft(v):
                soft_violations += 1
                if len(soft_samples) < 5:
                    soft_samples.append((str(lab_path), k, v))
            if not within01_hard(v):
                hard_violations += 1
                if len(hard_samples) < 3:
                    hard_samples.append((str(lab_path), k, v))

        # 경계(left/right/top/bot)
        left  = xc - w/2; right = xc + w/2
        top   = yc - h/2; bot   = yc + h/2
        for k, v in {"left":left, "right":right, "top":top, "bot":bot}.items():
            if not within01_soft(v):
                soft_violations += 1
                if len(soft_samples) < 5:
                    soft_samples.append((str(lab_path), k, v))
            if not within01_hard(v):
                hard_violations += 1
                if len(hard_samples) < 3:
                    hard_samples.append((str(lab_path), k, v))

        hist[c] += 1
    return hist

def class_hist_on_split(root: Path, split: str):
    img_dir = root/"images"/split
    lab_dir = root/"labels"/split
    assert img_dir.exists() and lab_dir.exists(), f"missing split dirs: {split}"
    imgs, labs, only_img, only_lab = check_files_exist(img_dir, lab_dir)
    print(f"[{split}] images={len(imgs)} labels={len(labs)}")
    if only_img:
        print(f"  [warn] labels missing for {len(only_img)} images (e.g., {list(sorted(list(only_img))[:3])})")
    if only_lab:
        print(f"  [warn] images missing for {len(only_lab)} labels (e.g., {list(sorted(list(only_lab))[:3])})")

    # 라벨 샘플링
    if len(labs) <= SAMPLE_LABELS_PER_SPLIT:
        sample = labs
    else:
        sample = random.sample(labs, SAMPLE_LABELS_PER_SPLIT)

    hist = Counter()
    for lp in sample:
        hist += validate_label_file(lp, img_dir)
    return hist, len(imgs), len(labs)

def check_yaml_files():
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"
    data = try_load_yaml(DATA_YAML)
    if isinstance(data, dict) and data.get("_minimal_ok"):
        print("[ok] data.yaml keys present (lenient parse)")
    else:
        # 정식 파싱 성공 시 필수 키/클래스명 확인
        assert "path" in data and "train" in data and "val" in data and "names" in data, "data.yaml missing keys"
        assert list(data["names"]) == CLASSES, f"data.yaml names mismatch: {data['names']}"
        print("[ok] data.yaml keys & class names")

    if CLASSES_YAML.exists():
        head = read_first_n_lines(CLASSES_YAML, n=1)
        if head:
            assert "title" in head[0] and "equation" in head[0], f"classes.yaml looks odd: {head[0]}"
        print("[ok] classes.yaml present")
    else:
        print("[warn] classes.yaml not found (optional)")

def text_box_count_for_image(stem: str, root: Path):
    for split in ["train","val"]:
        lp = root/"labels"/split/f"{stem}.txt"
        if lp.exists():
            n = 0
            with open(lp, "r", encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip(): continue
                    c = int(float(ln.split()[0]))
                    if c in TEXT_SET: n += 1
            return n
    return 0

def check_pairs_csv():
    if not PAIR_CSV.exists():
        print("[warn] pairs csv not found:", PAIR_CSV)
        return

    df = pd.read_csv(PAIR_CSV)
    need_cols = {"image_path","pair_id","src_idx","dst_idx","label"}
    missing = need_cols - set(df.columns)
    if missing:
        print(f"[warn] pairs missing columns: {missing}")
        return

    # 기본 통계
    total = len(df)
    pos = int((df["label"]==1).sum())
    neg = total - pos
    pos_ratio = pos / max(1, total)
    print(f"[pairs] rows={total}  pos={pos}  neg={neg}  pos_ratio={pos_ratio:.2%}")

    # 인덱스 무결성 & 존재성
    bad_rows = 0
    per_img_counts = []
    per_img_pos_ratio = []

    grp = df.groupby("image_path")
    for ipath, sub in grp:
        ip = Path(ipath)
        # 이미지/라벨 존재 체크(링크 경로 보정)
        img_ok = False
        for split in ["train","val"]:
            if (YOLO_BAL/"images"/split/ip.name).exists():
                img_ok = True; break
        if not img_ok:
            bad_rows += len(sub)
            continue

        # 텍스트 박스 개수
        T = text_box_count_for_image(ip.stem, YOLO_BAL)
        if T < 2:
            bad_rows += len(sub)
            continue

        # src/dst 범위
        ok = 0
        for r in sub.itertuples(index=False):
            a = int(r.src_idx); b = int(r.dst_idx)
            if 0 <= a < T and 0 <= b < T and a != b:
                ok += 1
        bad_rows += (len(sub) - ok)

        # per-image 분포 통계
        per_img_counts.append(len(sub))
        p = int((sub.label==1).sum())
        per_img_pos_ratio.append(p / max(1,len(sub)))

    ratio_bad = bad_rows / max(1, total)
    print(f"[pairs] invalid_index_rows={bad_rows} ({ratio_bad:.2%})")
    if ratio_bad >= 0.02:
        print("[warn] too many invalid src/dst indices (>2%). 확인 필요")

    # per-image 분포 요약
    if per_img_counts:
        print(f"[pairs] per-image pairs: mean={stats.mean(per_img_counts):.1f}, median={stats.median(per_img_counts):.1f}, max={max(per_img_counts)}")
    if per_img_pos_ratio:
        print(f"[pairs] per-image pos_ratio: mean={stats.mean(per_img_pos_ratio):.3f}, median={stats.median(per_img_pos_ratio):.3f}")

# ===== 실행 =====
print("== YAML / 메타 점검 ==")
check_yaml_files()

print("\n== train split 점검 ==")
h_tr, n_img_tr, n_lab_tr = class_hist_on_split(YOLO_BAL, "train")

print("\n== val split 점검 ==")
h_va, n_img_va, n_lab_va = class_hist_on_split(YOLO_BAL, "val")

# Split 요약
print("\n== split 요약 ==")
total_imgs = n_img_tr + n_img_va
val_ratio = (n_img_va / max(1,total_imgs)) if total_imgs else 0.0
print(f"images  train={n_img_tr}  val={n_img_va}  total={total_imgs}  (val_ratio={val_ratio:.2%})")
print(f"labels  train={n_lab_tr}  val={n_lab_va}")

# 클래스 히스토그램(샘플 기준)
def pretty_hist(h, title):
    total = sum(h.values())
    print(f"\n[{title}] class histogram (sampled)")
    for cid, name in enumerate(CLASSES):
        cnt = h.get(cid,0)
        pct = (cnt/max(1,total))*100
        print(f"  {name:9s} : {cnt:7d} ({pct:5.1f}%)")
    print(f"  total boxes (checked) : {total}")

pretty_hist(h_tr, "train")
pretty_hist(h_va, "val")

# 라벨 범위 이탈 요약
print("\n== 라벨 범위 이탈 요약 ==")
print(f"soft violations(±{TOL} 허용 초과): {soft_violations}")
if soft_samples:
    print("  e.g.", soft_samples[:3])
print(f"hard violations(±{HARD_TOL} 심각 초과): {hard_violations}")
if hard_samples:
    print("  e.g.", hard_samples)

# pairs 점검
print("\n== pairs.csv 점검 ==")
check_pairs_csv()

# STRICT 모드면 hard 위반 시 에러
if STRICT and hard_violations > 0:
    raise AssertionError(f"Hard violations detected: {hard_violations}")

print("\n✅ 검증 완료 (관대 모드). 리포트 수치를 확인하세요.")

# val파일 중복으로 인한 리더보드 성능누수 방지(중복제거)
from pathlib import Path
import shutil

root = Path("/content/workspace/dataset_balanced").resolve()
img_exts = {".png", ".jpg", ".jpeg"}
label_ext = ".txt"

VAL_IMG = root/"images/val"
VAL_LBL = root/"labels/val"
TRN_IMG = root/"images/train"
TRN_LBL = root/"labels/train"

# 1) val의 symlink를 "실파일"로 고정 (링크 타겟을 복사해와서 링크를 대체)
def materialize(dirp):
    fixed = broken = 0
    for p in dirp.iterdir():
        if not p.is_symlink():
            continue
        try:
            # 타겟 실존하면 그걸 복사하여 링크 대체
            target = p.resolve(strict=True)
            tmp = p.with_suffix(p.suffix + ".tmpcopy")
            shutil.copy2(target, tmp)
            p.unlink()             # 링크 삭제
            tmp.rename(p)          # 실제 파일로 교체
            fixed += 1
        except FileNotFoundError:
            # 이미 깨진 링크면 카운트만 (이후 중복 제거로 인한 것일 수 있음)
            broken += 1
    return fixed, broken

fix_i, br_i = materialize(VAL_IMG)
fix_l, br_l = materialize(VAL_LBL)
print(f"[VAL 고정] images: fixed={fix_i}, broken={br_i} | labels: fixed={fix_l}, broken={br_l}")

# 2) val 기준으로 train의 중복만 삭제 (val은 절대 손대지 않음)
val_stems = {p.stem for p in VAL_IMG.iterdir() if p.is_file() and p.suffix.lower() in img_exts}

# 이미지 중복 제거
del_img = 0
for p in TRN_IMG.iterdir():
    if p.is_file() and p.suffix.lower() in img_exts and p.stem in val_stems:
        p.unlink()
        del_img += 1

# 라벨 동기 삭제 + 고아 라벨 정리
del_lbl = 0
for p in TRN_LBL.iterdir():
    if p.is_file() and p.suffix.lower() == label_ext and p.stem in val_stems:
        p.unlink()
        del_lbl += 1

# 고아 라벨 제거(이미지가 사라져 남은 라벨)
train_img_stems = {p.stem for p in TRN_IMG.iterdir() if p.is_file() and p.suffix.lower() in img_exts}
orphan = [p for p in TRN_LBL.glob("*.txt") if p.stem not in train_img_stems]
for p in orphan:
    p.unlink()

print(f"[train 정리] 삭제된 중복 이미지={del_img}, 라벨={del_lbl}, 고아 라벨 추가 삭제={len(orphan)}")

# 3) (강력 권장) YOLO 캐시 제거
for p in [
    root/"labels/train.cache", root/"labels/val.cache",
    root/"images/train.cache", root/"images/val.cache",
]:
    try: p.unlink()
    except FileNotFoundError: pass

# 4) 최종 점검: val은 살아 있고(=깨진 링크 0), train/val 개수 확인
def count_files(d, exts):
    return sum(1 for p in d.iterdir() if p.is_file() and (p.suffix.lower() in exts if isinstance(exts, set) else p.suffix.lower()==exts))
def has_broken(dirp):
    return any(p.is_symlink() and not p.exists() for p in dirp.iterdir())

print(f"VAL images={count_files(VAL_IMG, img_exts)}, VAL labels={count_files(VAL_LBL, label_ext)}")
print("VAL 깨진 링크 존재?", has_broken(VAL_IMG) or has_broken(VAL_LBL))
print(f"TRAIN images={count_files(TRN_IMG, img_exts)}, TRAIN labels={count_files(TRN_LBL, label_ext)}")
print("✅ 준비 완료")

