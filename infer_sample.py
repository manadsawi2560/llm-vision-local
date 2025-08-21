import argparse, random, torch
from datasets import load_dataset
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

def load_model(ckpt):
    try:
        processor = BlipProcessor.from_pretrained(ckpt)
        model = BlipForConditionalGeneration.from_pretrained(ckpt)
    except Exception:
        ckpt = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(ckpt)
        model = BlipForConditionalGeneration.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model.to(device).eval(), device

def caption_image(model, processor, device, image):
    inputs = processor(images=image, text="a photo of", return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=30)
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/blip-flickr8k")
    parser.add_argument("--image", default=None, help="Path to your image")
    args = parser.parse_args()

    processor, model, device = load_model(args.ckpt)

    if args.image:
        img = Image.open(args.image).convert("RGB")
        print("Caption:", caption_image(model, processor, device, img))
        return

    # random from dataset
    ds = None
    for name in ["jxie/flickr8k", "mrfakename/flickr8k", "liamjamesk/flickr8k"]:
        try:
            ds = load_dataset(name)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Cannot load Flickr8k dataset.")

    split = "test" if "test" in ds else list(ds.keys())[0]
    ex = ds[split][random.randrange(len(ds[split]))]
    img = ex["image"] if isinstance(ex["image"], Image.Image) else Image.open(ex["image"]).convert("RGB")
    print("Caption:", caption_image(model, processor, device, img))
    img.show()

if __name__ == "__main__":
    main()