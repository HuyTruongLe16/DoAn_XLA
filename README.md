# Logo Detection & Brand Classification  
**YOLO + Feature Matching (SIFT / ORB)**

---

## 1. Giới thiệu
Đề tài xây dựng hệ thống **phát hiện và phân loại logo/thương hiệu trong ảnh**.  
Hệ thống nhận đầu vào là ảnh chụp thực tế hoặc ảnh web, tự động:
- Phát hiện **vị trí logo** trong ảnh (bounding box)
- Xác định **thương hiệu tương ứng** (label)

Đề tài kết hợp **Object Detection** và **Feature Matching** nhằm nâng cao độ chính xác và tính linh hoạt khi mở rộng thương hiệu mới.

---

## 2. Mục tiêu
- Phát hiện logo xuất hiện trong ảnh
- Xác định thương hiệu của logo
- Hiển thị trực quan:
  - Bounding box
  - Nhãn thương hiệu (label)
  - Điểm tương đồng (matching score)

---

## 3. Yêu cầu hệ thống
### 3.1. Đầu vào
- Ảnh `.jpg`, `.png`
- Ảnh có thể chứa một hoặc nhiều logo

### 3.2. Đầu ra
- Ảnh kết quả có:
  - Bounding box bao quanh logo
  - Label thương hiệu
- Có thể xuất kết quả ra màn hình hoặc giao diện demo

---

## 4. Kỹ thuật sử dụng

### 4.1. Object Detection
- **YOLOv8 (Ultralytics)**  
  - Phát hiện vị trí logo
  - Trả về bounding box + confidence score
- *(Mở rộng)* Faster R-CNN (mô tả trong báo cáo)

### 4.2. Feature Matching
- Trích xuất đặc trưng cục bộ bằng:
  - **SIFT** (chính xác cao)
  - **ORB** (nhanh hơn)
- So khớp đặc trưng bằng:
  - KNN Matching
  - Lowe’s Ratio Test
- So sánh với tập logo mẫu (reference logos)

### 4.3. Pipeline tổng thể
Ảnh đầu vào
↓
YOLO phát hiện logo (bounding box)
↓
Cắt vùng logo
↓
SIFT / ORB trích xuất đặc trưng
↓
So khớp với logo mẫu
↓
Gán nhãn thương hiệu + hiển thị kết quả 

---

## 5. Cấu trúc thư mục
DOAN_XLA/
│
├── models/
│ └── best.pt # Model YOLO đã train
│
├── reference_logos/
│ ├── adidas/
│ ├── nike/
│ └── coca_cola/
│
├── dataset/
│ └── data.yaml # Dataset YOLO
│
├── verifier_fixed.py # Feature Matching (SIFT/ORB)
├── detector_fixed.py # YOLO detector
├── pipeline_fixed.py # Pipeline tổng hợp
├── app_gradio_fixed.py # Giao diện demo
├── main_fixed.py # Chạy CLI
├── train_yolo_fixed.py # Train YOLO
└── README.md

---

## 6. Cài đặt môi trường

### 6.1. Cài Python
- Python **3.9 – 3.11** (khuyến nghị)

### 6.2. Cài thư viện
```bash
pip install ultralytics opencv-python numpy gradio
python train_yolo_fixed.py
runs/detect/logo_train/weights/best.pt
models/best.pt
python app_gradio_fixed.py
http://127.0.0.1:7860
