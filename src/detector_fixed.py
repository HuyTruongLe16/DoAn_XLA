# detector_fixed.py
from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO


@dataclass
class DetBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_id: int
    cls_name: str


class YOLODetector:
    def __init__(self, model_path="models/best.pt", conf=0.5, iou=0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, img_bgr: np.ndarray) -> List[DetBox]:
        results = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, verbose=False)
        r = results[0]
        boxes = []

        if r.boxes is None:
            return boxes

        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = r.names[cls_id]
            boxes.append(DetBox(x1, y1, x2, y2, conf, cls_id, cls_name))
        return boxes
