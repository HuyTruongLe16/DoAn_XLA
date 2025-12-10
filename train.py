from ultralytics import YOLO

# Khởi tạo mô hình (phiên bản small)
model = YOLO('yolov8s.pt')

# Bắt đầu huấn luyện mô hình, sử dụng file data.yaml
results = model.train(
    data='data.yaml', # Trỏ đến file data.yaml vừa tạo
    epochs=100,                     
    imgsz=640,                      
    batch=16,                       
    name='yolov8s_FlickrLogos32_Run1' 
)

# Sau khi huấn luyện, đánh giá trên tập test
metrics = model.val(split='test') 
print(f"mAP@50: {metrics.box.map50}, mAP@50-95: {metrics.box.map}")
