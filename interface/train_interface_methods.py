import gradio as gr
from ultralytics import YOLO

def interface_finetune():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')  # Load an official Detect model
    
def interface_train():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')  # Load an official Detect model