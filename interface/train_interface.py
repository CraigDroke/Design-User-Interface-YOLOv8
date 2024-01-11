import gradio as gr
from interface.defaults import shared_theme
from interface.train_interface_methods import interface_train

def build_train_interface():
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
        """
        # Training Interface for YOLOv8
        Train your own YOLOv8 model!
        """)
        
        with gr.Row() as finetune_row:
            is_finetune = gr.Checkbox(label="Finetune",info="Check this box if you want to finetune a model")
            is_offical_pretrained = gr.Checkbox(label="Official",info="Check this box if you want to train an official model",visible=True,interactive=True,value=True)
            custom_pretrained = gr.File(label="Pretrained Model Weights",file_count='single',type='binary',
                                    file_types=['.pt'],visible=True,show_label=True,interactive=True)
            offical_pretrained = gr.Dropdown(label="Pretrained Model",choices=["yolov8n.pt"],visible=True,interactive=True)
        with gr.Row() as dataset_row:
            is_official_dataset = gr.Checkbox(label="Official",info="Check this box if you want to use an official dataset",visible=True,interactive=True,value=True)
            custom_dataset = gr.File(label="Custom Dataset",file_count='single',type='binary',
                                    file_types=['.zip'],visible=True,show_label=True,interactive=True)
            official_dataset = gr.Dropdown(label="Dataset",choices=["coco128"],visible=True,interactive=True)
        with gr.Row() as output_row:
            logger = gr.Radio(choices=['WANDB', 'ClearML', 'Tensorboard'],value='WANDB',show_label=True,interactive=True,visible=True,
                              label="Logger",info="Choose 'W&B' if you want to use Weights & Biases, Choose 'Tensorboard' if you want to use Tensorboard")
        
        # Row for start & clear buttons
        with gr.Row() as buttons:
            start_but = gr.Button(value="Start")
        
        start_but.click(fn=interface_train,inputs=[is_finetune, official_dataset],outputs=[])
            
    return demo

if __name__== "__main__" :
    demo = build_train_interface()