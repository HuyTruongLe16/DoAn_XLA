"""
Author: hoangedu773
GitHub: https://github.com/hoangedu773
Date: 2025-12-16
Description: KNN Feature Matching module cho Logo Detection
"""

import cv2
import numpy as np
import glob
import os


class LogoMatcher:
    """
    Class thực hiện Feature Matching bằng SIFT + KNN
    Nhiệm vụ: So sánh logo đã detect với database ảnh mẫu
    """
    
    def __init__(self, reference_folder, algorithm='SIFT', n_features=1500):
        """
        Args:
            reference_folder: Thư mục chứa ảnh logo mẫu
            algorithm: 'SIFT' hoặc 'ORB'
            n_features: Số feature points
        """
        self.reference_folder = reference_folder
        self.algorithm = algorithm
        self.reference_db = {}
        
        # Khởi tạo detector
        if algorithm == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:  # ORB
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        print(f"✅ Khởi tạo {algorithm} detector")
        self._load_reference_database()
    
    def _load_reference_database(self):
        """Load và extract features từ ảnh mẫu"""
        if not os.path.exists(self.reference_folder):
            os.makedirs(self.reference_folder)
            print(f"⚠️ Chưa có thư mục {self.reference_folder}")
            return
        
        print(f"📂 Đang load ảnh mẫu từ {self.reference_folder}...")
        
        for img_path in glob.glob(os.path.join(self.reference_folder, '*.*')):
            try:
                logo_name = os.path.splitext(os.path.basename(img_path))[0]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Extract features
                keypoints, descriptors = self.detector.detectAndCompute(img, None)
                
                if descriptors is not None and len(descriptors) > 5:
                    self.reference_db[logo_name] = descriptors
                    
            except Exception as e:
                print(f"❌ Lỗi load {img_path}: {e}")
        
        print(f"✅ Đã load {len(self.reference_db)} logo: {list(self.reference_db.keys())}")
    
    def _calculate_knn_score(self, des_query, des_reference, ratio=0.75):
        """
        Tính điểm matching bằng Lowe's ratio test
        
        Args:
            des_query: Descriptors từ ảnh cần nhận diện
            des_reference: Descriptors từ ảnh mẫu
            ratio: Ngưỡng Lowe's ratio (0.7-0.8)
        
        Returns:
            Số lượng good matches
        """
        if des_query is None or des_reference is None:
            return 0
        
        try:
            # KNN matching với k=2
            matches = self.matcher.knnMatch(des_query, des_reference, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio * n.distance:
                        good_matches.append(m)
            
            return len(good_matches)
            
        except Exception as e:
            print(f"⚠️ Lỗi KNN: {e}")
            return 0
    
    def match(self, cropped_logo_bgr, threshold=10):
        """
        Nhận diện logo bằng KNN matching
        
        Args:
            cropped_logo_bgr: Ảnh logo đã crop (BGR format)
            threshold: Ngưỡng số good matches tối thiểu
        
        Returns:
            (logo_name, confidence_score)
        """
        # Chuyển sang grayscale
        if len(cropped_logo_bgr.shape) == 3:
            gray = cv2.cvtColor(cropped_logo_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_logo_bgr
        
        # Tăng tương phản (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Extract features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) < 2:
            return "Unknown", 0
        
        # So sánh với từng logo trong database
        best_name = "Unknown"
        best_score = 0
        
        for logo_name, ref_descriptors in self.reference_db.items():
            score = self._calculate_knn_score(descriptors, ref_descriptors)
            
            if score > best_score:
                best_score = score
                best_name = logo_name
        
        # Kiểm tra threshold
        if best_score >= threshold:
            return best_name, best_score
        else:
            return "Unknown", best_score
    
    def reload_database(self):
        """Reload lại database (khi có thêm ảnh mẫu mới)"""
        self.reference_db.clear()
        self._load_reference_database()


# ====================
# DEMO USAGE
# ====================
if __name__ == "__main__":
    # Test với ảnh mẫu
    matcher = LogoMatcher(reference_folder='reference', algorithm='SIFT')
    
    # Test với 1 ảnh
    test_img = cv2.imread('test.jpg')
    if test_img is not None:
        logo_name, score = matcher.match(test_img)
        print(f"🎯 Kết quả: {logo_name} (score: {score})")
    else:
        print("⚠️ Không tìm thấy file test.jpg")
