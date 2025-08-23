import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import joblib
import json
from paddleocr import PaddleOCR

# -----------------------------
# 1. Khởi tạo PhoBERT
# -----------------------------
MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 2. Khởi tạo OCR (không dùng cls nữa)
# -----------------------------
ocr = PaddleOCR(use_angle_cls=True, lang="vi")

# -----------------------------
# 3. Hàm OCR ảnh
# -----------------------------
def extract_text(image_path: str) -> str:
    result = ocr.ocr(image_path)  # KHÔNG truyền cls
    lines = []
    if result:
        for res in result:
            for line in res:
                lines.append(line[1][0])
    return " ".join(lines)

# -----------------------------
# 4. Hàm sinh vector từ văn bản
# -----------------------------
def embed_text(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(768)  # phòng trường hợp OCR ra rỗng
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

# -----------------------------
# 5. Build dataset từ thư mục
# -----------------------------
def build_dataset(dataset_dir: str):
    X, y = [], []
    labels = sorted(os.listdir(dataset_dir))
    for idx, label in enumerate(labels):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, file)
                print(f" Đang OCR: {img_path}")
                text = extract_text(img_path)
                vec = embed_text(text)
                X.append(vec)
                y.append(idx)
    return np.array(X), np.array(y), labels

# -----------------------------
# 6. Train & lưu model
# -----------------------------
if __name__ == "__main__":
    dataset_dir = "dataset"  # thư mục dữ liệu
    X, y, labels = build_dataset(dataset_dir)

    print(" Đang train Logistic Regression...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    joblib.dump(clf, "doc_classifier.pkl")
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(" Training done! Saved model & labels.")
    
