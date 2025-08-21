from __future__ import annotations
import argparse, json, pickle
import numpy as np
from PIL import Image
import tensorflow as tf

from src.model import build_inception_encoder

def greedy_decode(model, tok, max_len, feat_vec):
    inv_index = {i: w for w, i in tok.word_index.items()}
    inv_index[0] = "<pad>"
    start_id = tok.word_index.get("startseq")
    end_id = tok.word_index.get("endseq")
    seq = [start_id]
    for _ in range(max_len - 1):
        inp = np.array([seq + [0]*(max_len - len(seq))], dtype=np.int32)
        preds = model.predict([feat_vec[None, :], inp], verbose=0)[0]
        next_id = int(np.argmax(preds[len(seq)-1]))
        if next_id == 0:
            break
        seq.append(next_id)
        if next_id == end_id:
            break
    words = [inv_index.get(i, "<unk>") for i in seq]
    words = [w for w in words if w not in {"startseq", "endseq", "<pad>"}]
    return " ".join(words)

def main(image_path: str):
    encoder, preprocess = build_inception_encoder()
    img = Image.open(image_path).convert("RGB").resize((299, 299))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess(arr)
    feat = encoder(arr[None, ...], training=False).numpy()[0]

    with open("artifacts/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    with open("artifacts/max_len.json", "r", encoding="utf-8") as f:
        max_len = json.load(f)["max_len"]

    try:
        model = tf.keras.models.load_model("artifacts/ckpt/model.keras")
    except Exception:
        model = tf.keras.models.load_model("artifacts/final_model.keras")

    print(greedy_decode(model, tok, max_len, feat))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()
    main(args.image)