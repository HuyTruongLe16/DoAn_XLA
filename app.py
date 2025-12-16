import cv2
import numpy as np
import glob
import os
import gradio as gr
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
MODEL_PATH = 'models/best.pt'    # Đường dẫn model đã train
REF_FOLDER = 'reference'         # Thư mục chứa ảnh mẫu

# Khởi tạo thuật toán
print("⚙️ Đang khởi tạo YOLO và ORB...")
try:
    model = YOLO(MODEL_PATH)
except:
    print(f"⚠️ Chưa thấy file {MODEL_PATH}. Đang dùng tạm yolov8n.pt demo.")
    model = YOLO('yolov8n.pt')

orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

reference_features = {}

# ==========================================
# 2. HÀM LOAD DỮ LIỆU MẪU
# ==========================================
def load_reference_logos():
    reference_features.clear()
    if not os.path.exists(REF_FOLDER):
        os.makedirs(REF_FOLDER)
        print(f"⚠️ Thư mục '{REF_FOLDER}' chưa có ảnh. Hãy copy ảnh logo mẫu vào đó!")
        return

    print(f"🔄 Đang học các logo mẫu từ thư mục '{REF_FOLDER}'...")
    for img_path in glob.glob(os.path.join(REF_FOLDER, '*.*')):
        try:
            name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            kp, des = orb.detectAndCompute(img, None)
            
            if des is not None and len(des) > 5:
                reference_features[name] = des
        except Exception as e:
            print(f"Lỗi file {img_path}: {e}")

    print(f"✅ Đã thuộc làu {len(reference_features)} logo: {list(reference_features.keys())}")

# Load ngay khi chạy app
load_reference_logos()

# ==========================================
# 3. CÁC HÀM XỬ LÝ (MODULAR)
# ==========================================
def get_orb_descriptors(img_bgr):
    """Trích xuất đặc trưng từ ảnh crop"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    kp, des = orb.detectAndCompute(gray, None)
    return des

def calculate_knn_score(des_input, des_reference):
    """Tính điểm khớp giữa 2 mẫu vân tay"""
    if des_input is None or des_reference is None:
        return 0
    try:
        matches = bf.knnMatch(des_input, des_reference, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return len(good_matches)
    except:
        return 0

def identify_logo(cropped_logo_bgr, threshold=4):
    """Hàm định danh chính"""
    des_input = get_orb_descriptors(cropped_logo_bgr)
    if des_input is None or len(des_input) < 2:
        return "Unknown", 0

    best_name = "Unknown"
    best_score = 0

    for name, ref_des in reference_features.items():
        score = calculate_knn_score(des_input, ref_des)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score

# ==========================================
# 4. HÀM CHÍNH (PIPELINE)
# ==========================================
def predict_pipeline(input_image, conf_threshold):
    if input_image is None: return None

    # Gradio trả về RGB, OpenCV cần BGR
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated = img_bgr.copy()

    # 1. YOLO Detect
    results = model.predict(source=img_bgr, conf=conf_threshold, iou=0.5, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 2. Crop & Identify
        cropped_logo = img_bgr[y1:y2, x1:x2]
        if cropped_logo.size == 0: continue

        match_name, match_score = identify_logo(cropped_logo)

        # 3. Draw
        label = f"{match_name} ({match_score})"
        color = (0, 255, 0) if match_name != "Unknown" else (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ==========================================
# 5. GIAO DIỆN WEB (GRADIO)
# ==========================================
if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict_pipeline,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Slider(0, 1, value=0.5, label="Confidence")
        ],
        outputs=gr.Image(label="Result"),
        title="YOLOv8 + ORB Local App",
        description="Chạy trực tiếp trên VS Code."
    )
    
 
    demo.launch(inbrowser=True)