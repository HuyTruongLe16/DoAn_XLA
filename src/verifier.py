# verifier_fixed.py
# Feature Matching (SIFT / ORB) để phân loại thương hiệu logo

import cv2
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MatchResult:
    brand: str
    score: int
    method: str
    ratio: float
    min_good: int


class LogoMatcher:
    def __init__(self, method="SIFT", ratio=0.75, min_good=8, max_features=2000):
        self.method = method.upper()
        self.ratio = ratio
        self.min_good = min_good
        self.max_features = max_features
        self.refs: Dict[str, List[np.ndarray]] = {}

        if self.method == "SIFT":
            self.extractor = cv2.SIFT_create()
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.method == "ORB":
            self.extractor = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError("Method must be SIFT or ORB")

    def load_reference_logos(self, folder="reference_logos"):
        self.refs.clear()
        for brand in os.listdir(folder):
            brand_path = os.path.join(folder, brand)
            if not os.path.isdir(brand_path):
                continue

            desc_list = []
            for img_path in glob.glob(os.path.join(brand_path, "*")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.equalizeHist(img)
                _, des = self.extractor.detectAndCompute(img, None)
                if des is not None:
                    desc_list.append(des)

            if desc_list:
                self.refs[brand] = desc_list

        if not self.refs:
            raise RuntimeError("Không load được logo mẫu")

    def match_logo(self, cropped_bgr):
        if cropped_bgr is None or cropped_bgr.size == 0:
            return MatchResult("Unknown", 0, self.method, self.ratio, self.min_good)

        gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (220, 220))
        _, des = self.extractor.detectAndCompute(gray, None)

        if des is None:
            return MatchResult("Unknown", 0, self.method, self.ratio, self.min_good)

        best_brand = "Unknown"
        best_score = 0

        for brand, ref_descs in self.refs.items():
            brand_best = 0
            for ref_des in ref_descs:
                matches = self.matcher.knnMatch(des, ref_des, k=2)
                good = [m for m, n in matches if m.distance < self.ratio * n.distance]
                brand_best = max(brand_best, len(good))
            if brand_best > best_score:
                best_score = brand_best
                best_brand = brand

        if best_score >= self.min_good:
            return MatchResult(best_brand, best_score, self.method, self.ratio, self.min_good)
        return MatchResult("Unknown", best_score, self.method, self.ratio, self.min_good)
