import gradio as gr
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
from ultralytics import YOLO

dirname = os.path.dirname(__file__)

dirname = dirname.split("Clinic_2")[0] + "Clinic_2"

def count_train_folders(directory):
    count = 0
    for folder in os.listdir(directory):
        if "train" in folder:
            count += 1
    return count

count = count_train_folders(os.path.join(dirname, 'runs', 'detect'))

def interface_login(logger,pretrained,dataset,epochs):
    if logger == 'WANDB':
        result = wandb.login()
        #subprocess.run(key, shell=True)
        
        if result:
            gr.Info("Logged in to WANDB")
        else:
            gr.Warning("Failed to log in to WANDB")
        interface_train_wandb('yolov8', pretrained, dataset, epochs, 640)
    elif logger == 'Tensorboard':
        gr.Info("Logged in to Tensorboard")
        interface_train_tensorboard(pretrained, dataset, epochs, 640)
        
    

# def interface_finetune():
#     # Load a pretrained YOLOv8n model
#     model = YOLO('yolov8n.pt')  # Load an official Detect model
#     return model
    
def interface_train(model_name, dataset, epochs, imgsz=640):
    epochs=int(epochs)
    print("In train function")
    #model_name = os.path.basename(model_name)
    model = YOLO(model_name)
    # if is_finetune:
    #     model = YOLO('yolov8n.pt')
    model.train(data=dataset, epochs=epochs, imgsz=imgsz)
    
def interface_train_wandb(project_name, model_name, dataset_name, epochs, imgsz=640):
    # Step 1: Initialize a Weights & Biases run
    wandb.init(project=project_name, job_type="training")

    model = YOLO(f"{model_name}")


    # Step 3: Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Step 4: Train and Fine-Tune the Model
    model.train(project=project_name, data=dataset_name, epochs=epochs, imgsz=imgsz)

    # Step 5: Validate the Model
    model.val()

    # Step 7: Finalize the W&B Run
    wandb.finish()

def interface_train_tensorboard(model_name, dataset_name, epochs, imgsz=640):
    model = YOLO(f"{model_name}")
    
    model.train(data=dataset_name, epochs=epochs, imgsz=imgsz)
    tb_dir = os.path.join(dirname,"runs","detect",f"train{count}")
    command = f"python -m tensorboard.main --logdir={tb_dir}"
    subprocess.run(command, shell=True)
    gr.Info("Results logged to Tensorboard")