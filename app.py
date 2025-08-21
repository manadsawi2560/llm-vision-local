from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch, io, os

app = FastAPI(title="BLIP Flickr8k – Local Captioning API")

CKPT_DIR = os.environ.get("BLIP_CKPT", "outputs/blip-flickr8k")

# โหลด checkpoint ที่ฝึกไว้ ถ้าไม่พบจะ fallback เป็นโมเดลฐาน
try:
    processor = BlipProcessor.from_pretrained(CKPT_DIR)
    model = BlipForConditionalGeneration.from_pretrained(CKPT_DIR)
except Exception:
    BASE = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(BASE)
    model = BlipForConditionalGeneration.from_pretrained(BASE)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

@app.get("/", response_class=HTMLResponse)
def root():
    return open("static/index.html", "r", encoding="utf-8").read()

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    inputs = processor(images=image, text="a photo of", return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return {"caption": text}