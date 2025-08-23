import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
import joblib
import json
from transformers import AutoTokenizer, AutoModel
from paddleocr import PaddleOCR
import re

# Load mô hình phân loại
clf = joblib.load("doc_classifier.pkl")
with open("labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

# Load PhoBERT
MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Khởi tạo OCR (đã bật angle classifier)
ocr_paddle = PaddleOCR(use_angle_cls=True, lang="vi")

# Hàm embed text với PhoBERT
def embed_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# Giao diện Streamlit
st.title(" Hệ thống nhận dạng & phân loại giấy tờ")

uploaded_file = st.file_uploader(" Tải lên ảnh giấy tờ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Tiền xử lý ảnh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(thresh, -1, kernel)

    # OCR với PaddleOCR
    text_paddle = ocr_paddle.ocr(np.array(image)) 
    lines = []
    for res in text_paddle:
        for line in res:
            lines.append(line[1][0])
    text_joined = " ".join(lines)

    # OCR với Tesseract
    text_tess = pytesseract.image_to_string(sharpen, lang="vie")

    # Kết hợp text
    text_final = (text_joined + " " + text_tess).strip()
    text_final = text_final.upper()
    text_final = re.sub(r"\s+", " ", text_final)

    # Hiển thị kết quả
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Ảnh gốc & xử lý")
        st.image(image, caption="Ảnh gốc", use_column_width=True)
        st.image(sharpen, caption="Ảnh sau tiền xử lý", use_column_width=True, channels="GRAY")
    with col2:
        st.subheader(" Văn bản OCR")
        st.text_area("Kết quả OCR", text_final, height=300)

    # Phân loại văn bản
    if text_final.strip():
        vec = embed_text(text_final)
        pred = clf.predict([vec])[0]
        st.success(f" Loại giấy tờ dự đoán: **{labels[pred]}**")
