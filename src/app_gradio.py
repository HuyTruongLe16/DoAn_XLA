# app_gradio_fixed.py
import cv2
import gradio as gr
from pipeline_fixed import LogoDetectionPipeline

pipeline = LogoDetectionPipeline()

def predict(img):
    if img is None:
        return None
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = pipeline.run(img_bgr)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(),
    title="Logo Detection & Brand Classification",
    description="YOLO phát hiện bounding box + SIFT phân loại thương hiệu"
).launch()
