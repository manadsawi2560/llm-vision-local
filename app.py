from __future__ import annotations
import io, json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf

from src.model import build_inception_encoder

ARTIFACTS = Path("artifacts")

app = Flask(__name__, static_folder="static", static_url_path="")

encoder, preprocess = build_inception_encoder()

with open(ARTIFACTS / "tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)
with open(ARTIFACTS / "max_len.json", "r", encoding="utf-8") as f:
    max_len = json.load(f)["max_len"]

try:
    model = tf.keras.models.load_model(ARTIFACTS / "ckpt" / "model.keras")
except Exception:
    model = tf.keras.models.load_model(ARTIFACTS / "final_model.keras")

inv_index = {i: w for w, i in tok.word_index.items()}
inv_index[0] = "<pad>"
START_ID = tok.word_index.get("startseq")
END_ID = tok.word_index.get("endseq")

def greedy_decode(feat_vec: np.ndarray) -> str:
    seq = [START_ID]
    for _ in range(max_len - 1):
        inp = np.array([seq + [0]*(max_len - len(seq))], dtype=np.int32)
        preds = model.predict([feat_vec[None, :], inp], verbose=0)[0]
        next_id = int(np.argmax(preds[len(seq)-1]))
        if next_id == 0:
            break
        seq.append(next_id)
        if next_id == END_ID:
            break
    words = [inv_index.get(i, "<unk>") for i in seq]
    words = [w for w in words if w not in {"startseq", "endseq", "<pad>"}]
    return " ".join(words)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file field"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((299, 299))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess(arr)
    feat = encoder(arr[None, ...], training=False).numpy()[0]

    caption = greedy_decode(feat)
    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)