import gradio as gr
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

def interface_finetune():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')  # Load an official Detect model
    return model
    
def interface_train(is_fintune=False, dataset=None, epochs=2, imgsz=640):
    model = YOLO('yolov8n.yaml')
    if is_fintune:
        model = interface_finetune()
    results = model.train(data=dataset, epochs=epochs, imgsz=imgsz)
    
def interface_train_wandb():
    # Step 1: Initialize a Weights & Biases run
    wandb.init(project="ultralytics", job_type="training")

    # Step 2: Define the YOLOv8 Model and Dataset
    model_name = "yolov8n"
    dataset_name = "coco128.yaml"
    model = YOLO(f"{model_name}.pt")

    # Step 3: Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Step 4: Train and Fine-Tune the Model
    model.train(project="ultralytics", data=dataset_name, epochs=5, imgsz=640)

    # Step 5: Validate the Model
    model.val()

    # Step 6: Perform Inference and Log Results
    model(["Images\Craig.jpg", "Images\WalterWhite.jpg"])

    # Step 7: Finalize the W&B Run
    wandb.finish()