from __future__ import annotations
import subprocess
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

print("[i] Downloading Flickr8k from Kaggle (adityajn105/flickr8k) ...")
cmd = ["kaggle", "datasets", "download", "-d", "adityajn105/flickr8k", "-p", str(RAW), "--unzip"]
subprocess.check_call(cmd)
print("[✓] Done. Verify folders: data/raw/Flickr8k_Dataset and data/raw/Flickr8k_text")