from __future__ import annotations
import argparse
from pathlib import Path

import tensorflow as tf

from src.dataset import make_tf_dataset, load_tokenizer, ARTIFACTS
from src.model import build_caption_model

def main(epochs: int, batch_size: int):
    tok, max_len = load_tokenizer()
    vocab_size = max(tok.word_index.values()) + 1

    print(f"[i] Vocab size: {vocab_size}, Max len: {max_len}")

    model = build_caption_model(vocab_size=vocab_size, max_len=max_len)

    train_ds = make_tf_dataset("train", batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset("val", batch_size=batch_size, shuffle=False)

    ckpt_dir = ARTIFACTS / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_dir / "model.keras"),
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
    es = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt_cb, es]
    )

    model.save(ARTIFACTS / "final_model.keras")
    print("[✓] Training complete. Model saved to artifacts/.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    main(args.epochs, args.batch_size)