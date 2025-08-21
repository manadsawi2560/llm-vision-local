from __future__ import annotations
import argparse
from src.dataset import prepare_tokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_freq", type=int, default=5, help="minimum frequency to keep a word")
    args = ap.parse_args()
    prepare_tokenizer(min_freq=args.min_freq)
    print("Tokenizer + splits prepared (artifacts/, data/splits/)")