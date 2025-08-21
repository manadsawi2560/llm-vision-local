# src/dataset.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import json
import pickle
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .utils import clean_caption, read_lines, ensure_dir

# ---- พาธหลัก ----
RAW_DIR = Path("data/raw")
TXT_DIR = RAW_DIR / "Flickr8k_text"
FEATURE_DIR = Path("data/features")
SPLIT_DIR = Path("data/splits")
ARTIFACTS = Path("artifacts")

# ---- รูป: พยายามเดาหลายแบบ (Flickr8k_Dataset, images/, หรือ RAW_DIR) ----
_IMG_CANDIDATES = [
    RAW_DIR / "Flickr8k_Dataset",
    RAW_DIR / "images",
    RAW_DIR,
    Path("data/images"),
]
def _first_existing_dir(cands):
    for c in cands:
        if c.exists() and c.is_dir():
            return c
    return cands[0]
IMG_BASE = _first_existing_dir(_IMG_CANDIDATES)

# ---- โหมดข้อมูล & ไฟล์ CSV ที่รองรับ ----
CSV_CANDIDATES = [
    RAW_DIR / "caption.txt",
    RAW_DIR / "captions.txt",
    RAW_DIR / "captions.csv",
    RAW_DIR / "caption.csv",
]

START_TOKEN = "startseq"
END_TOKEN = "endseq"

# ---------------------------
# ช่วยตรวจจับรูปแบบข้อมูล
# ---------------------------
def _has_flickr8k_text() -> bool:
    req = [
        TXT_DIR / "Flickr8k.token.txt",
        TXT_DIR / "Flickr_8k.trainImages.txt",
        TXT_DIR / "Flickr_8k.devImages.txt",
        TXT_DIR / "Flickr_8k.testImages.txt",
    ]
    return all(p.exists() for p in req)

def _find_caption_csv() -> Path | None:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    return None

def _load_csv_mapping(csv_path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    # รองรับ header: image,caption
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or not any(h.lower() in ("image", "img", "filename") for h in header):
            f.seek(0)
            reader = csv.reader(f)
            header = ["image", "caption"]
        h_lower = [h.lower() for h in header]
        try:
            i_img = h_lower.index("image")
        except ValueError:
            i_img = h_lower.index("filename")
        i_cap = h_lower.index("caption")

        for row in reader:
            if not row or len(row) <= max(i_img, i_cap):
                continue
            fn = row[i_img].strip()
            cap = clean_caption(row[i_cap])
            if fn and cap:
                mapping[fn].append(cap)
    return mapping

# ---------------------------
# ส่วนสำหรับ splits และ captions
# ---------------------------
def load_filename_lists() -> Tuple[List[str], List[str], List[str]]:
    """
    ถ้ามี Flickr8k_text แบบมาตรฐาน → ใช้ไฟล์ split เดิม
    ถ้าไม่มี แต่มี CSV (image,caption) → ถ้า data/splits มีแล้ว ใช้เลย
      มิฉะนั้น สร้าง train/val/test ด้วยการสุ่ม 80/10/10 (โดยยึดรายชื่อรูปจาก CSV)
    """
    if _has_flickr8k_text():
        train = read_lines(TXT_DIR / "Flickr_8k.trainImages.txt")
        val = read_lines(TXT_DIR / "Flickr_8k.devImages.txt")
        test = read_lines(TXT_DIR / "Flickr_8k.testImages.txt")
        return train, val, test

    csv_path = _find_caption_csv()
    if not csv_path:
        raise FileNotFoundError(
            "ไม่พบ Flickr8k_text และไม่พบไฟล์ CSV (caption.txt/captions.txt/captions.csv) ใน data/raw/"
        )

    # ถ้ามี splits เดิมก็ใช้
    train_p = SPLIT_DIR / "train.txt"
    val_p = SPLIT_DIR / "val.txt"
    test_p = SPLIT_DIR / "test.txt"
    if train_p.exists() and val_p.exists() and test_p.exists():
        return read_lines(train_p), read_lines(val_p), read_lines(test_p)

    # สร้าง splits ใหม่จาก CSV
    cap_map = _load_csv_mapping(csv_path)
    images = sorted(cap_map.keys())
    if not images:
        raise RuntimeError("ไฟล์ CSV ไม่มีข้อมูลรูป")

    random.seed(1337)
    random.shuffle(images)
    n = len(images)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    ensure_dir(SPLIT_DIR)
    (SPLIT_DIR / "train.txt").write_text("\n".join(train), encoding="utf-8")
    (SPLIT_DIR / "val.txt").write_text("\n".join(val), encoding="utf-8")
    (SPLIT_DIR / "test.txt").write_text("\n".join(test), encoding="utf-8")

    return train, val, test

def build_caption_dict() -> Dict[str, List[str]]:
    """
    คืน mapping: filename -> [caption, ...]
    รองรับทั้งรูปแบบ Flickr8k_text และ CSV (image,caption)
    """
    if _has_flickr8k_text():
        token_file = TXT_DIR / "Flickr8k.token.txt"
        mapping: Dict[str, List[str]] = defaultdict(list)
        for line in read_lines(token_file):
            try:
                key, caption = line.split("\t")
                filename = key.split("#")[0]
                mapping[filename].append(clean_caption(caption))
            except Exception:
                continue
        return mapping

    csv_path = _find_caption_csv()
    if not csv_path:
        raise FileNotFoundError(
            "ไม่พบ Flickr8k_text และไม่พบไฟล์ CSV (caption.txt/captions.txt/captions.csv) ใน data/raw/"
        )
    return _load_csv_mapping(csv_path)

def save_tokenizer(tokenizer: Tokenizer, max_len: int) -> None:
    ensure_dir(ARTIFACTS)
    with open(ARTIFACTS / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    (ARTIFACTS / "max_len.json").write_text(json.dumps({"max_len": max_len}), encoding="utf-8")

def load_tokenizer() -> Tuple[Tokenizer, int]:
    with open(ARTIFACTS / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    max_len = json.loads((ARTIFACTS / "max_len.json").read_text(encoding="utf-8"))["max_len"]
    return tok, max_len

def _apply_min_freq(captions: List[str], min_freq: int) -> List[str]:
    cnt = Counter()
    for c in captions:
        for w in c.split():
            if w not in {START_TOKEN, END_TOKEN}:
                cnt[w] += 1
    filtered = []
    for c in captions:
        toks = []
        for w in c.split():
            if w in {START_TOKEN, END_TOKEN}:
                toks.append(w)
            else:
                toks.append(w if cnt[w] >= min_freq else "<unk>")
        filtered.append(" ".join(toks))
    return filtered

def prepare_tokenizer(min_freq: int = 5) -> None:
    train, _, _ = load_filename_lists()
    caption_map = build_caption_dict()

    captions = []
    missing = 0
    for fname in train:
        caps = caption_map.get(fname, [])
        if not caps:
            missing += 1
            continue
        for c in caps:
            captions.append(f"{START_TOKEN} {c} {END_TOKEN}")

    if missing > 0 and missing / max(1, len(train)) > 0.05:
        raise RuntimeError(
            "Captions missing/invalid too often for training images. ตรวจพาธรูป/ไฟล์ CSV/splits อีกครั้ง"
        )

    captions = _apply_min_freq(captions, min_freq=min_freq)

    tok = Tokenizer(oov_token="<unk>", filters="", lower=False)
    tok.fit_on_texts(captions)
    max_len = max(len(c.split()) for c in captions)

    save_tokenizer(tok, max_len)

def img_path(filename: str) -> Path:
    """
    คืน Path ของไฟล์รูป โดยลองหลาย base dir ตามที่พบจริง
    """
    for base in [IMG_BASE, RAW_DIR / "images", RAW_DIR, Path("data/images")]:
        p = base / filename
        if p.exists():
            return p
    # ถ้าไม่เจอจริง ๆ ก็คืนพาธตามค่าแรก (ให้สคริปต์ขึ้น warning ตอน extract_features)
    return (RAW_DIR / "Flickr8k_Dataset" / filename)

def feature_path(filename: str) -> Path:
    ensure_dir(FEATURE_DIR)
    return FEATURE_DIR / (Path(filename).stem + ".npy")

def captions_for(filename: str, cap_map: Dict[str, List[str]]) -> List[str]:
    caps = cap_map.get(filename, [])
    return [f"{START_TOKEN} {c} {END_TOKEN}" for c in caps]

def sequences_from_captions(tok: Tokenizer, captions: List[str], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    seqs = tok.texts_to_sequences(captions)
    X = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    y = np.copy(X)
    y[:, :-1] = X[:, 1:]
    y[:, -1] = 0
    return X, y

def make_tf_dataset(split: str, batch_size: int = 64, shuffle: bool = True) -> tf.data.Dataset:
    assert split in {"train", "val", "test"}
    tok, max_len = load_tokenizer()
    cap_map = build_caption_dict()

    filelist = read_lines(SPLIT_DIR / f"{split}.txt")
    if not filelist:
        raise FileNotFoundError(f"ไม่พบไฟล์ splits สำหรับ {split}.txt — โปรดรัน prepare_tokenizer ก่อน")

    X_feats, X_seqs, Y = [], [], []
    for fname in filelist:
        fpath = feature_path(fname)
        if not fpath.exists():
            # ข้ามถ้ายังไม่ได้ extract features ของไฟล์นี้
            continue
        feats = np.load(fpath)
        for cap in captions_for(fname, cap_map):
            X_seq, y_seq = sequences_from_captions(tok, [cap], max_len)
            X_feats.append(feats)
            X_seqs.append(X_seq[0])
            Y.append(y_seq[0])

    if len(Y) == 0:
        raise RuntimeError("ไม่พบตัวอย่างข้อมูล (ตรวจว่าได้รัน extract_features.py ครบ และ mapping รูป–แคปชันถูกต้อง)")

    X_feats = np.array(X_feats, dtype=np.float32)
    X_seqs = np.array(X_seqs, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(((X_feats, X_seqs), Y))
    if shuffle and split == "train":
        ds = ds.shuffle(buffer_size=min(10000, len(Y)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
