from ultralytics import YOLO

if __name__ == "__main__":
    # Load YOLO pretrained
    model = YOLO("yolov8n.pt")

    # Train với dataset trong đồ án
    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="logo_train"
    )

    print(" Train xong! Model nằm trong runs/detect/logo_train/weights/")
