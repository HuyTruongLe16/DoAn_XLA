# DoAn_XLA
# Đồ án: Phân loại & Định vị Logo/Thương hiệu trong Ảnh

## 1. Mục tiêu
Xây dựng hệ thống phát hiện vị trí (Bounding Box) và phân loại 32 thương hiệu logo sử dụng kỹ thuật Object Detection.

## 2. Kỹ thuật Chính
* **Object Detection:** YOLOv8
* **Dữ liệu:** FlickrLogos-32 (Nguồn: Roboflow)

## 3. Cách chạy
1.  Cài đặt môi trường: `pip install ultralytics torch`
2.  Tải dữ liệu FlickrLogos-32 (theo cấu hình trong data.yaml).
3.  Chạy huấn luyện: `python train.py`
