from __future__ import annotations
from pathlib import Path
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import tensorflow as tf

from src.dataset import load_tokenizer, build_caption_dict, feature_path
from src.utils import read_lines
from src.model import build_caption_model

ARTIFACTS = Path("artifacts")

def load_best_model():
    tok, max_len = load_tokenizer()
    vocab_size = max(tok.word_index.values()) + 1
    model = build_caption_model(vocab_size=vocab_size, max_len=max_len)
    ckpt = ARTIFACTS / "ckpt" / "model.keras"
    if ckpt.exists():
        model = tf.keras.models.load_model(ckpt)
    else:
        model = tf.keras.models.load_model(ARTIFACTS / "final_model.keras")
    return model, tok, max_len

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

def main():
    nltk.download("punkt", quiet=True)
    model, tok, max_len = load_best_model()
    cap_map = build_caption_dict()
    test_files = read_lines(Path("data/splits/test.txt"))

    references = []
    hypotheses = []
    for fname in test_files:
        fpath = feature_path(fname)
        if not fpath.exists():
            continue
        feat = np.load(fpath)
        hyp = greedy_decode(model, tok, max_len, feat)
        refs = [c.split() for c in cap_map.get(fname, [])]
        references.append(refs)
        hypotheses.append(hyp.split())

    smoothie = SmoothingFunction().method4
    bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)

    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

if __name__ == "__main__":
    main()