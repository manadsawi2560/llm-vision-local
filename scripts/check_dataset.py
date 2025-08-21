# scripts/check_dataset.py
from pathlib import Path
from src.dataset import _find_caption_csv, _load_csv_mapping, IMG_BASE

def main():
    csv_path = _find_caption_csv()
    assert csv_path and csv_path.exists(), "ไม่พบ caption.txt/captions.csv ใน data/raw/"
    cap_map = _load_csv_mapping(csv_path)
    missing = []
    for fn in list(cap_map.keys())[:2000]:
        p = IMG_BASE / fn
        if not p.exists():
            missing.append(str(p))
    print(f"Images listed in CSV: {len(cap_map):,}")
    print(f"IMG_BASE: {IMG_BASE}")
    if missing:
        print(f"Missing files (show up to 20): {missing[:20]}")
    else:
        print("OK: All referenced images exist under IMG_BASE.")

if __name__ == "__main__":
    main()
