
from ultralytics import YOLO
import glob
import gradio as gr
import numpy as np
import cv2
import os
import shutil



model = YOLO('yolov8s.pt')

orb = cv2.ORB_create(nfeatures=1500)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) 

reference_features = {}

def load_reference_logos(folder='/content/reference'):
    reference_features.clear()
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"⚠️ Thư mục '{folder}' chưa có ảnh. Hãy upload ảnh mẫu vào!")
        return

    print(f"🔄 Đang load ảnh mẫu từ {folder} (ORB)...")

    for img_path in glob.glob(folder + '/*.*'):
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # --- DÙNG ORB ---
        kp, des = orb.detectAndCompute(img, None)

        # Chỉ nhận nếu tìm thấy đủ điểm đặc trưng
        if des is not None and len(des) > 5:
            reference_features[name] = des

    print("✅ Đã load xong logo mẫu (ORB):", list(reference_features.keys()))


load_reference_logos('/reference')

# ==========================================
# 4. HÀM SO KHỚP (LOGIC ORB + KNN)
# ==========================================
def match_logo(cropped_logo_bgr, threshold=4): 
    gray = cv2.cvtColor(cropped_logo_bgr, cv2.COLOR_BGR2GRAY)

    # Tăng tương phản (Vẫn giữ lại vì tốt cho cả ORB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # --- DÙNG ORB ---
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or len(des) < 2:
        return "Unknown", 0

    best_name = "Unknown"
    best_score = 0

    for name, ref_des in reference_features.items():
        try:
            # KNN Match với k=2 (Tìm 2 điểm giống nhất)
            matches = bf.knnMatch(des, ref_des, k=2)

            # --- LOWE'S RATIO TEST CHO ORB ---
            # ORB lỏng hơn SIFT nên ta dùng tỉ lệ 0.75 (thay vì 0.7 của SIFT)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            score = len(good_matches)

            if score > best_score:
                best_score = score
                best_name = name
        except:
            continue

    if best_score >= threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score


def yolo_orb_predict(input_image, conf_threshold):
    if input_image is None:
        return None

    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated = img_bgr.copy()

    # YOLO Predict
    results = model.predict(source=img_bgr, conf=conf_threshold, iou=0.5, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cropped_logo = img_bgr[y1:y2, x1:x2]
        if cropped_logo.size == 0:
            continue

        # Gọi hàm match_logo (ORB)
        match_name, match_score = match_logo(cropped_logo)

        label = f"{match_name} ({match_score})"
        color = (0, 255, 0) if match_name != "Unknown" else (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Khởi tạo Gradio
demo = gr.Interface(
    fn=yolo_orb_predict,
    inputs=[
        gr.Image(type="numpy", label="Upload ảnh"),
        gr.Slider(0, 1, value=0.5, label="Confidence Threshold")
    ],
    outputs=gr.Image(label="Kết quả"),
    title="YOLOv8 + ORB Logo Detection",
    description="Phiên bản sử dụng thuật toán ORB."
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)

