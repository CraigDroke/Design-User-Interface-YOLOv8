from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb
import gradio as gr

def example_train():
    wandb.login()
    # Step 1: Initialize a Weights & Biases run
    wandb.init(project="ultralytics", job_type="training")

    # Step 2: Define the YOLOv8 Model and Dataset
    model_name = "yolov8n"
    dataset_name = "coco128.yaml"
    model = YOLO(f"{model_name}.pt")

    # Step 3: Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Step 4: Train and Fine-Tune the Model
    model.train(project="ultralytics", data=dataset_name, epochs=2, imgsz=640)

    # Step 5: Validate the Model
    model.val()

    # Step 6: Perform Inference and Log Results
    model(["examples\\training\Images\Craig.jpg", "examples\\training\Images\WalterWhite.jpg"])

    # Step 7: Finalize the W&B Run
    wandb.finish()


if __name__ == "__main__":
    title = "JoJoGAN"
    description = "Gradio Demo for JoJoGAN: One Shot Face Stylization. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."

    demo = gr.Interface(
        example_train,
        gr.Image(type="pil"),
        gr.Image(type="file"),
        title=title,
        description=description
    )

    demo.launch(share=True)
    demo.integrate(wandb=wandb)