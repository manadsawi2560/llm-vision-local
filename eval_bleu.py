import argparse, torch
from typing import List
from PIL import Image
from datasets import load_dataset
from transformers import BlipForConditionalGeneration, BlipProcessor
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

def get_refs(example) -> List[str]:
    refs = []
    if "captions" in example and example["captions"]:
        for c in example["captions"]:
            if isinstance(c, dict) and "raw" in c:
                refs.append(str(c["raw"]))
            elif isinstance(c, str):
                refs.append(c)
    if not refs and "sentences" in example and example["sentences"]:
        for s in example["sentences"]:
            if isinstance(s, dict):
                r = s.get("raw") or s.get("sentence")
                if r:
                    refs.append(str(r))
            elif isinstance(s, str):
                refs.append(s)
    if not refs and "caption" in example and example["caption"]:
        refs = [str(example["caption"])]
    if not refs and "text" in example and example["text"]:
        refs = [str(example["text"])]
    if not refs:
        refs = ["a photo of something"]
    return refs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/blip-flickr8k")
    parser.add_argument("--split", default="validation", choices=["train","validation","test"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max_new", type=int, default=30)
    args = parser.parse_args()

    # load model
    try:
        processor = BlipProcessor.from_pretrained(args.ckpt)
        model = BlipForConditionalGeneration.from_pretrained(args.ckpt)
    except Exception:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # load dataset
    ds = None
    for name in ["jxie/flickr8k","mrfakename/flickr8k","liamjamesk/flickr8k"]:
        try:
            ds = load_dataset(name)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Cannot load Flickr8k dataset.")

    split = args.split if args.split in ds else list(ds.keys())[0]
    N = min(args.limit, len(ds[split]))

    hyps, refs = [], []
    smooth = SmoothingFunction().method1

    for i in tqdm(range(N), desc=f"Evaluating on {split}"):
        ex = ds[split][i]
        img = ex.get("image") or ex.get("img")
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        inputs = processor(images=img, text="a photo of", return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=args.max_new)
        gen = processor.batch_decode(out, skip_special_tokens=True)[0]

        hyps.append(gen.split())
        refs.append([r.split() for r in get_refs(ex)])

    from math import isfinite
    b1 = corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=smooth)
    b2 = corpus_bleu(refs, hyps, weights=(0.5,0.5,0,0), smoothing_function=smooth)
    b3 = corpus_bleu(refs, hyps, weights=(1/3,1/3,1/3,0), smoothing_function=smooth)
    b4 = corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

    print(f"BLEU-1: {b1:.4f}")
    print(f"BLEU-2: {b2:.4f}")
    print(f"BLEU-3: {b3:.4f}")
    print(f"BLEU-4: {b4:.4f}")

if __name__ == "__main__":
    # ensure nltk data
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except Exception:
        import nltk
        nltk.download("punkt")
    main()