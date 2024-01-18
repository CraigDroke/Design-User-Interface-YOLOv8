import gradio as gr
from ultralytics import YOLO
import numpy as np

def interface_detect(source,weights,thres,pretrained,use_custom):
    # Load a pretrained YOLOv8n model
    print(pretrained)
    if use_custom:
        model = YOLO(pretrained)
    else:
        if not weights.endswith(".pt"):
            weights = weights + ".pt"
        model = YOLO(weights)  # Load an official Detect model
    if isinstance(source, gr.Video):
        print("To be added")
    elif isinstance(source, np.ndarray):
        results = model.predict(source=source,conf=thres/100)
        return results[0].plot()
    else:
        raise ValueError("Invalid source type")