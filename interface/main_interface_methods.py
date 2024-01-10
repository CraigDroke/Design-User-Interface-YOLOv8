
import gradio as gr
from ultralytics import YOLO
import numpy as np

def interface_detect(source):
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')  # Load an official Detect model
    if isinstance(source, gr.Video):
        print("To be added")
    elif isinstance(source, np.ndarray):
        results = model.predict(source)
        return results[0].plot()
    else:
        raise ValueError("Invalid source type")