# pipeline_fixed.py
import cv2
from detector_fixed import YOLODetector
from verifier import LogoMatcher


class LogoDetectionPipeline:
    def __init__(self, model_path="models/best.pt", ref_dir="reference_logos"):
        self.detector = YOLODetector(model_path)
        self.matcher = LogoMatcher(method="SIFT")
        self.matcher.load_reference_logos(ref_dir)

    def run(self, img_bgr):
        annotated = img_bgr.copy()
        boxes = self.detector.detect(img_bgr)

        for box in boxes:
            crop = img_bgr[box.y1:box.y2, box.x1:box.x2]
            result = self.matcher.match_logo(crop)

            color = (0, 255, 0) if result.brand != "Unknown" else (0, 0, 255)
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)

            label = f"{result.brand} ({result.score})"
            cv2.putText(
                annotated, label,
                (box.x1, box.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return annotated
