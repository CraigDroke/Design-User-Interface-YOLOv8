from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
source = "C:\\Users\\modon\\Downloads\\evil.png"
results = model.predict(source,visualize=True)