# KNN Feature Matching Module

**Tác giả:** hoangedu773  
**Phần việc:** Feature Matching (KNN)

---

## 📋 Mô tả

Module này thực hiện **Feature Matching** sử dụng thuật toán **KNN** (K-Nearest Neighbors) kết hợp với **Lowe's Ratio Test** để nhận diện logo.

### Quy trình hoạt động:

```
Logo được YOLO detect ──> SIFT/ORB extract features ──> KNN matching ──> Tên logo
```

---

## 🔧 Cài đặt

```bash
pip install opencv-python numpy
```

---

## 📂 Cấu trúc thư mục

```
DoAn_XLA/
├── matching.py          ← Module KNN matching (PHẦN CỦA BẠN)
├── app.py               ← Gradio UI
├── train.py             ← Train YOLO
├── reference/           ← Thư mục ảnh logo mẫu
│   ├── cocacola.png
│   ├── pepsi.png
│   └── ...
└── models/
    └── best.pt          ← Model YOLO đã train
```

---

## 🚀 Cách sử dụng

### 1. Thêm ảnh logo mẫu

Upload ảnh logo vào thư mục `reference/`:
- Tên file = tên logo (VD: `cocacola.png`, `nike.jpg`)
- Nên dùng ảnh nền trắng, logo rõ nét

### 2. Sử dụng trong code

```python
from matching import LogoMatcher

# Khởi tạo
matcher = LogoMatcher(
    reference_folder='reference',
    algorithm='SIFT',  # hoặc 'ORB'
    n_features=1500
)

# Nhận diện logo
import cv2
logo_img = cv2.imread('cropped_logo.jpg')
logo_name, score = matcher.match(logo_img, threshold=10)

print(f"Logo: {logo_name}, Score: {score}")
```

### 3. Tích hợp với YOLO

Xem file `app.py` để biết cách kết hợp YOLO + KNN matching.

---

## ⚙️ Tham số

| Tham số | Mô tả | Giá trị mặc định |
|---------|-------|------------------|
| `algorithm` | SIFT hoặc ORB | `'SIFT'` |
| `n_features` | Số feature points | `1500` |
| `threshold` | Ngưỡng số good matches | `10` |
| `ratio` | Lowe's ratio test | `0.75` |

---

## 📊 Đánh giá

- **SIFT:** Chính xác hơn, chậm hơn
- **ORB:** Nhanh hơn, kém chính xác hơn
- **Lowe's ratio:** 0.7-0.8 (càng thấp càng strict)

---

## 🐛 Troubleshooting

### Lỗi: "Đã load 0 logo"
→ Kiểm tra thư mục `reference/` có ảnh chưa

### Kết quả luôn "Unknown"
→ Giảm `threshold` hoặc thêm ảnh mẫu chất lượng cao

### Nhận diện sai
→ Thử đổi `algorithm='ORB'` sang `'SIFT'`

---

## 📝 Ghi chú

- Module này độc lập, có thể dùng riêng hoặc tích hợp vào app
- Đã tối ưu với CLAHE để tăng độ tương phản
- Sử dụng Lowe's ratio test để lọc good matches
