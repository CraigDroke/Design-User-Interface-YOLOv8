from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)   # train yolov8n on COCO128 for 100 epochs
results = model.train(data='coco128.yaml', epochs=2, imgsz=640)  # train yolov8n on COCO128 for 2 epochs