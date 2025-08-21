from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from src.model import build_inception_encoder
from src.dataset import load_filename_lists, img_path, feature_path, FEATURE_DIR

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

encoder, preprocess = build_inception_encoder()

def load_and_preprocess_image(p: Path) -> tf.Tensor:
    img = Image.open(p).convert("RGB").resize((299, 299))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess(arr)
    return tf.convert_to_tensor(arr)

def main():
    train, val, test = load_filename_lists()
    all_files = list(dict.fromkeys(train + val + test))
    print(f"[i] Total unique images: {len(all_files)}")

    for fname in tqdm(all_files, desc="Extracting features"):
        out = feature_path(fname)
        if out.exists():
            continue
        p = img_path(fname)
        if not p.exists():
            print(f"[warn] missing image: {p}")
            continue
        x = load_and_preprocess_image(p)
        x = tf.expand_dims(x, 0)
        feats = encoder(x, training=False).numpy()[0]
        np.save(out, feats.astype(np.float32))

    print("[✓] Features saved to", FEATURE_DIR)

if __name__ == "__main__":
    main()