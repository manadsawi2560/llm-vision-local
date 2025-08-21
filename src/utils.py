from __future__ import annotations
import re
import string
from pathlib import Path

def clean_caption(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[-_]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)