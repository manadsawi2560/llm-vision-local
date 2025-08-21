import argparse, os
from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATASET_CANDIDATES = [
    "jxie/flickr8k",       # ตัวเลือกหลัก
    "mrfakename/flickr8k", # สำรอง
    "liamjamesk/flickr8k", # สำรอง
]

def try_load_flickr8k():
    last_err = None
    for ds_name in DATASET_CANDIDATES:
        try:
            ds = load_dataset(ds_name)
            # หากมีแค่ train ให้แตก validation/test เอง
            if "train" in ds and len(ds.keys()) == 1:
                split = ds["train"].train_test_split(test_size=0.1, seed=42)
                valtest = split["test"].train_test_split(test_size=0.5, seed=42)
                ds = {"train": split["train"], "validation": valtest["train"], "test": valtest["test"]}
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot load Flickr8k. Last error: {last_err}")

def pick_caption(example: Dict) -> str:
    # รองรับคีย์หลายรูปแบบ
    if "caption" in example and example["caption"]:
        return str(example["caption"])
    if "captions" in example and example["captions"]:
        c = example["captions"][0]
        if isinstance(c, dict) and "raw" in c:
            return str(c["raw"])
        return str(c)
    if "sentences" in example and example["sentences"]:
        s = example["sentences"][0]
        if isinstance(s, dict):
            return str(s.get("raw") or s.get("sentence") or s)
        return str(s)
    if "text" in example and example["text"]:
        return str(example["text"])
    return "a photo of something"

class Flickr8kTorch(Dataset):
    def __init__(self, hf_split, processor, max_len=40):
        self.ds = hf_split
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex.get("image") or ex.get("img")
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        text = pick_caption(ex).strip()
        inputs = self.processor(images=img, text=text, return_tensors="pt",
                                padding="max_length", max_length=self.max_len, truncation=True)
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = item["input_ids"].clone()
        return item

@dataclass
class DataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        b = {}
        for k in ["input_ids", "attention_mask", "pixel_values", "labels"]:
            b[k] = torch.stack([f[k] for f in features])
        return b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--out_dir", default="outputs/blip-flickr8k")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    print(">> Loading dataset ...")
    ds = try_load_flickr8k()

    print(">> Loading model & processor ...")
    processor = BlipProcessor.from_pretrained(args.model_id)
    model = BlipForConditionalGeneration.from_pretrained(args.model_id)

    train_set = Flickr8kTorch(ds["train"], processor, max_len=args.max_len)
    val_set   = Flickr8kTorch(ds.get("validation", ds["test"]), processor, max_len=args.max_len)

    fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=DataCollator(),
    )

    print(">> Training ...")
    trainer.train()

    print(">> Saving ...")
    trainer.save_model(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"✅ Done. Model saved to {args.out_dir}")

if __name__ == "__main__":
    main()