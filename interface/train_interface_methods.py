import gradio as gr
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

def interface_login(logger, args):
    if logger == 'WANDB':
        result = False
        wandb_key = args[0]
        if (wandb_key is not None) & isinstance(wandb_key, str):
            try:
                result = wandb.login(key=wandb_key,relogin=True)
            except:
                gr.Warning("Issue with the WANDB key")
        else:
            gr.Warning("Issue with the WANDB key")
        if result:
            gr.Info("Logged in to WANDB")
        else:
            gr.Warning("Failed to log in to WANDB")
    elif logger == 'ClearML':
        pass
    elif logger == 'Tensorboard':
        pass

def interface_finetune():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')  # Load an official Detect model
    return model
    
def interface_train(is_fintune=False, dataset=None, epochs=2, imgsz=640):
    model = YOLO('yolov8n.yaml')
    if is_fintune:
        model = interface_finetune()
    results = model.train(data=dataset, epochs=epochs, imgsz=imgsz)
    
def interface_train_wandb(project_name, model_name, dataset_name, epochs=2, imgsz=640):
    # Step 1: Initialize a Weights & Biases run
    wandb.init(project=project_name, job_type="training")

    model = YOLO(f"{model_name}.pt")

    # Step 3: Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Step 4: Train and Fine-Tune the Model
    model.train(project=project_name, data=dataset_name, epochs=epochs, imgsz=imgsz)

    # Step 5: Validate the Model
    model.val()

    # # Step 6: Perform Inference and Log Results
    # model(["Images\Craig.jpg", "Images\WalterWhite.jpg"])

    # Step 7: Finalize the W&B Run
    wandb.finish()