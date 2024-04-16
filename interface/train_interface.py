import gradio as gr
from interface.defaults import shared_theme
from interface.train_interface_methods import interface_train, interface_login
import random
# Define global variables
pretrained = gr.File()
dataset = gr.File()
global training_epochs
def build_train_interface():
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
        """
        # Training Interface for YOLOv8
        Train your own YOLOv8 model!
        """)
        

        with gr.Row() as finetune_row:
            
            #is_finetune = gr.Checkbox(label="Finetune",info="Check this box if you want to finetune a model")
            epochs = gr.Slider(label="Epochs", minimum=1, maximum=10, step=1, value=2, visible=True, interactive=True)
            #epochs.release(identity, inputs=[epochs, state], outputs=[number, state], api_name="predict")
            
            custom_pretrained = gr.File(label="Custom Model Weights",file_count='single',type='filepath',
                                    file_types=['.pt'],visible=True,show_label=True,interactive=True)
            official_pretrained = gr.Dropdown(label="Official Model",choices=["yolov8n.pt","yolov8s.pt","yolov9e.pt"],visible=True,interactive=True)
            
        with gr.Row() as dataset_row:
            custom_dataset = gr.File(label="Custom Dataset",file_count='single',type='filepath',
                                    file_types=['.zip'],visible=True,show_label=True,interactive=True)
            official_dataset = gr.Dropdown(label="Official Dataset",choices=["coco128.yaml", "coco8.yaml"],visible=True,interactive=True)

        # Row for start & clear buttons
        with gr.Row() as buttons:
            start_but = gr.Button(value="Start")
        with gr.Accordion("Logger Options") as login_accordion:
            logger = gr.Radio(choices=['WANDB', 'Tensorboard'],value='WANDB',show_label=True,interactive=True,visible=True,
                              label="Logger",info="Choose which logger to use")
            key = gr.Textbox(label="WANDB Key",placeholder="Enter WANDB Key",visible=True,interactive=True)
            login_but = gr.Button(value="Start with Logger")
            
        
        
        
        # Define update functions for pretrained model and dataset
        def update_pretrained(custom_pretrained, official_pretrained):
            global pretrained
            if custom_pretrained is not None:
                pretrained = custom_pretrained
            elif official_pretrained is not None:
                pretrained = official_pretrained

            print(f"Model: {pretrained}")

        def update_dataset(custom_dataset, official_dataset):
            global dataset
            if custom_dataset is not None:
                dataset = custom_dataset
            elif official_dataset is not None:
                dataset = official_dataset

            print(f"Dataset: {dataset}")

        


        
        # Connect file upload and dropdown change events to update functions
        custom_pretrained.upload(fn=update_pretrained, inputs=[custom_pretrained, official_pretrained], outputs=[])
        custom_dataset.upload(fn=update_dataset, inputs=[custom_dataset, official_dataset], outputs=[])
        official_pretrained.change(fn=update_pretrained, inputs=[custom_pretrained, official_pretrained], outputs=[])
        official_dataset.change(fn=update_dataset, inputs=[custom_dataset, official_dataset], outputs=[])
        #epochs.change(fn = clean_epochs, inputs=[epochs], outputs=[])
        # Define click actions
        start_but.click(fn=lambda: interface_train(model_name=pretrained, dataset=dataset, epochs=random.randint(1,10)), inputs=[], outputs=[])
        login_but.click(fn=lambda: interface_login(logger=logger,pretrained=pretrained, dataset=dataset, epochs=epochs,key=key), inputs = [] ,outputs=[])

    return demo

if __name__ == "__main__":
    demo = build_train_interface()

