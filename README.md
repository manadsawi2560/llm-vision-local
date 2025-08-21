# Image Captioning (Flickr8k) — CNN Encoder + LSTM Decoder (TensorFlow/Keras)

โปรเจกต์นี้สาธิตการ **สร้างโมเดลอธิบายภาพ (Image Captioning)** โดยใช้สถาปัตยกรรมแบบ **Encoder–Decoder**:
- **Encoder**: ใช้ **InceptionV3 (ImageNet)** สกัด **feature 2048 มิติ** จากภาพ  
- **Decoder**: ใช้ **LSTM** ทำนายคำอธิบายทีละคำ (sequence-to-sequence)  
- **โครงพื้นฐานที่คุณมี**: `data/raw/images/` + `data/raw/caption.txt` (CSV: `image,caption`)  


มีสคริปต์ครบตั้งแต่เตรียมข้อมูล → สกัดฟีเจอร์ → เทรน → ประเมิน → ทดสอบ → ดีพลอย API (Flask)

---

## สถาปัตยกรรม
1) **Image Encoder**: InceptionV3 (include_top=False, pooling='avg') → feature vector 2048  
2) **Initial States**: อนุมานค่า **h, c** ของ LSTM จาก feature ด้วย Dense  
3) **Text Decoder**: Embedding + LSTM (return_sequences=True) → TimeDistributed(Dense|softmax)  
4) **Teacher Forcing** ตอนเทรน (เลื่อน target 1 ตำแหน่ง)

---

## โครงสร้างโปรเจกต์

```plaintext
project-root/
├─ src/
│ ├─ model.py # โมเดล Encoder (InceptionV3) + Decoder (LSTM)
│ ├─ dataset.py # รองรับ caption CSV + auto split + tf.data pipelines
│ └─ utils.py # clean caption, io helper
├─ scripts/
│ ├─ prepare_captions.py # สร้าง tokenizer + สุ่ม splits (80/10/10 เมื่อเป็น CSV)
│ ├─ extract_features.py # ดึงฟีเจอร์ภาพ (InceptionV3) → data/features/*.npy
│ ├─ train.py # เทรนโมเดล LSTM decoder
│ ├─ evaluate.py # คำนวณ BLEU บน test set
│ └─ check_dataset.py # (ตัวเลือก) ตรวจ mapping image ↔ caption
├─ data/
│ ├─ raw/
│ │ ├─ images/ # โฟลเดอร์ภาพ (ถ้าใช้ CSV)
│ │ └─ caption.txt # CSV header: image,caption
│ ├─ features/ # *.npy จากการสกัดฟีเจอร์ (จะถูกสร้าง)
│ └─ splits/ # train.txt / val.txt / test.txt (จะถูกสร้าง)
├─ artifacts/ # tokenizer.pkl, max_len.json, ckpt/ (จะถูกสร้าง)
├─ static/
│ └─ index.html # หน้าเว็บสำหรับอัปโหลดรูปและดูคำบรรยาย
├─ app.py # Flask API
├─ predict_cli.py # พยากรณ์ผ่าน CLI
└─ requirements.txt
```

---

## ข้อกำหนดระบบ & Dependencies

- **Python**: แนะนำ **3.9–3.10**  
- **TensorFlow**: โครงการนี้ตั้งค่าตามสาย **TF 2.10.x** (เพื่อรองรับ Windows GPU ได้)  
  - CPU: ติดตั้ง `tensorflow==2.10.1` ได้เลย  
  - GPU (ทางเลือก): ต้องมี **CUDA 11.2** + **cuDNN 8.1** ที่ติดตั้งถูกต้อง  
- ไลบรารีอื่น ๆ: `numpy==1.23.5`, `scikit-learn==1.3.2`, `matplotlib==3.7.3`, `pandas`, `Pillow`, `tqdm`, `nltk`, `flask`, `waitress`, `protobuf<4`, `kaggle` (ถ้าจะดาวน์โหลดชุดข้อมูลแบบเดิม)

> **หมายเหตุ**: ถ้าคุณมีโปรเจกต์อื่นใน environment เดียวกัน อาจเกิด version conflict ได้ แนะนำให้สร้าง venv แยกเฉพาะโปรเจกต์นี้

---

## การเตรียม Environment (Windows CMD)

> จากโฟลเดอร์ **project-root/** ของโปรเจกต์นี้

```bat
:: 1) สร้างและเปิดใช้งาน virtualenv (ใช้ Python 3.9–3.10)
python -m venv C:\venvs\cap_tf210
C:\venvs\cap_tf210\Scripts\activate.bat

:: 2) ติดตั้ง dependencies (ตาม requirements สาย TF 2.10.x)
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

:: (ตรวจสอบ TensorFlow)
python - << "PY"
import tensorflow as tf
print("TF", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
PY
```
##  รัน Pipeline
1) เตรียม tokenizer + สร้าง splits
```bash
python -m scripts.prepare_captions --min_freq 5
```
2) สกัดฟีเจอร์ภาพ (InceptionV3 → 2048-D)
```bash
python -m scripts.extract_features
```

3) เทรนโมเดล
```bash
python -m scripts.train --epochs 20 --batch_size 64
```
บันทึก checkpoint ที่ artifacts/ckpt/model.keras และโมเดลสุดท้าย artifacts/final_model.keras

4) ประเมิน BLEU บน test set
```bash
python -m scripts.evaluate
```
5) ทดสอบพยากรณ์รูปเดี่ยว (CLI)
```bash
python predict_cli.py --image data\raw\images\1000268201_693b08cb0e.jpg
```
6) ดีพลอย API (Flask) + หน้าเว็บ
```bash
python app.py
:: เปิด http://127.0.0.1:5000 แล้วอัปโหลดรูปทดสอบในหน้าเว็บ
```

## แนวทางปรับแต่ง / ปรับปรุงคุณภาพ

- เพิ่มรอบเทรน/ลด min_freq/เพิ่ม batch size (ถ้า GPU มี VRAM พอ)

- ใช้ Beam Search แทน greedy decoding

- ปรับ Embed dim / LSTM units ใน src/model.py

- ทำ fine-tune encoder (unfreeze บาง block) เมื่อเทรนช่วงท้าย

- เพิ่ม data augmentation ตอนสกัดฟีเจอร์ (หากแก้ให้สกัดทุก epoch)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)