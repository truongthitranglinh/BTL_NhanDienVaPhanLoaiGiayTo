
# Hệ thống Nhận dạng & Phân loại Giấy tờ bằng AI

## Giới thiệu
Dự án này xây dựng hệ thống **nhận dạng và phân loại giấy tờ** (chứng minh thư, hộ chiếu, giấy phép lái xe, bằng cấp, hóa đơn, v.v) dựa trên:

- **OCR (Nhận dạng ký tự quang học)**: trích xuất văn bản từ ảnh giấy tờ bằng **PaddleOCR** và **Tesseract**.  
- **PhoBERT**: mô hình ngôn ngữ tiếng Việt để sinh vector đặc trưng cho văn bản.  
- **Logistic Regression**: phân loại văn bản giấy tờ dựa trên embedding PhoBERT.  
- **Streamlit**: giao diện web trực quan để người dùng tải ảnh, xem kết quả OCR và loại giấy tờ dự đoán.

## Kiến trúc hệ thống
1. **Tiền xử lý ảnh**: Chuyển xám, làm mờ, nhị phân hóa, sharpen bằng OpenCV.  
2. **OCR**: Kết hợp **PaddleOCR** (chính) và **Tesseract** (bổ sung).  
3. **Embedding**: Văn bản OCR được mã hóa thành vector 768 chiều bởi PhoBERT.  
4. **Phân loại**: Logistic Regression dự đoán loại giấy tờ.  
5. **Giao diện**: Streamlit hiển thị ảnh, văn bản OCR và kết quả phân loại.

## Cấu trúc thư mục
```
.
├── app.py               # Ứng dụng Streamlit (demo)
├── train.py             # Script huấn luyện mô hình phân loại
├── dataset/             # Dữ liệu huấn luyện (ảnh chia theo thư mục nhãn)
│   ├── cccd/
│   ├── ho_chieu/
│   └── bang_lai/
├── doc_classifier.pkl   # Mô hình Logistic Regression đã huấn luyện
├── labels.json          # Danh sách nhãn giấy tờ
├── requirements.txt     # Thư viện cần cài
```

## Cài đặt môi trường
### 1. Clone repo & tạo môi trường
```bash
git clone <repo_url>
cd <repo_name>
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

Nội dung `requirements.txt` gợi ý:
```
streamlit
opencv-python
numpy
pytesseract
Pillow
torch
transformers
scikit-learn
joblib
paddleocr
```

### 3. Cài đặt Tesseract
- **Windows**: [Download Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)  
- **Linux/Mac**:  
  ```bash
  sudo apt install tesseract-ocr
  ```

## Huấn luyện mô hình
1. Chuẩn bị dữ liệu theo cấu trúc:
   ```
   dataset/
   ├── cccd/        # ảnh Căn cước công dân
   ├── ho_chieu/    # ảnh Hộ chiếu
   └── bang_lai/    # ảnh Bằng lái xe
   ```

2. Chạy script train:
   ```bash
   python train.py
   ```
   - Kết quả:
     - `doc_classifier.pkl`: mô hình Logistic Regression đã huấn luyện.
     - `labels.json`: danh sách nhãn tương ứng.

## Chạy ứng dụng
```bash
streamlit run app.py
```

- Tải ảnh giấy tờ (`.jpg`, `.jpeg`, `.png`).  
- Hệ thống sẽ:
  - Tiền xử lý ảnh.  
  - Nhận dạng văn bản bằng OCR.  
  - Sinh vector embedding bằng PhoBERT.  
  - Dự đoán loại giấy tờ và hiển thị kết quả.  

## Ví dụ kết quả
- **Ảnh gốc & tiền xử lý**: hiển thị song song.  
- **Văn bản OCR**: nội dung trích xuất từ ảnh.  
- **Loại giấy tờ dự đoán**: Ví dụ: *Căn cước công dân*.

## Hướng phát triển
- Bổ sung thêm nhiều loại giấy tờ (bằng cấp, hóa đơn, hợp đồng...).  
- Tinh chỉnh PhoBERT để tăng độ chính xác.  
- Tích hợp API REST để triển khai thực tế.  
- Hỗ trợ nhiều ngôn ngữ ngoài tiếng Việt.  
