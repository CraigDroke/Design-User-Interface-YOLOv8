import gradio as gr
import time
from datetime import timedelta

from interface.defaults import shared_theme
from interface.train_interface_methods import interface_train, interface_login, file_check

class TrainInterface():
    def __init__(self):
        self.demo = None
        self.api_key = None
        self.accordion_open = False
        self.file_type = 'filepath'
        with gr.Blocks(theme=shared_theme) as demo:
            gr.Markdown(
            """
            # Training Interface for YOLOv8
            Train your own YOLOv8 model!
            """)
            with gr. Accordion("Finetune Settings",open=self.accordion_open):
                with gr.Row() as finetune_row:
                    is_finetune = gr.Checkbox(label="Finetune",info="Check this box if you want to finetune a model")
                    is_offical_pretrained = gr.Checkbox(label="Official",info="Check this box if you want to train an official model",visible=True,interactive=True,value=True)
                    custom_pretrained = gr.File(label="Pretrained Model Weights",file_count='single',type=self.file_type,
                                            file_types=['.pt'],visible=True,show_label=True,interactive=True)
                    offical_pretrained = gr.Dropdown(label="Pretrained Model",choices=["yolov8n.pt"],visible=True,interactive=True)
            with gr.Accordion("Dataset Settings",open=self.accordion_open):
                with gr.Row() as dataset_row:
                    is_official_dataset = gr.Checkbox(label="Official",info="Check this box if you want to use an official dataset",visible=True,interactive=True,value=True)
                    custom_dataset = gr.File(label="Custom Dataset",file_count='single',type=self.file_type,
                                            file_types=['.zip','.yaml'],visible=True,show_label=True,interactive=True)
                    official_dataset = gr.Dropdown(label="Dataset",choices=["coco128"],visible=True,interactive=True)

            with gr.Accordion("Logger Options",open=self.accordion_open) as login_accordion:
                use_logger = gr.Checkbox(label="Use Logger",info="Check this box if you want to use a logger",visible=True,interactive=True,value=False)
                logger = gr.Radio(choices=['WANDB', 'ClearML', 'Tensorboard'],value='WANDB',show_label=True,interactive=True,visible=True,
                                label="Logger",info="Choose which logger to use")
                wandb_key = gr.Textbox(label="WANDB Key",placeholder="Enter WANDB Key",visible=True,interactive=True)
                login_but = gr.Button(value="Login")

            # Row for start & clear buttons
            with gr.Row() as buttons:
                start_but = gr.Button(value="Start")
            
            def string_from_textbox(textbox):
                self.api_key = textbox
                
            wandb_key.change(fn=string_from_textbox,inputs=[wandb_key],outputs=[])
            
            def logger_login(use_logger, logger):
                if use_logger:
                    interface_login(logger, [self.api_key])
                else:
                    gr.Warning("Not using logger, so no need to login")
            
            def timer():
                start_time = time.monotonic()
            
            start_but.click(fn=interface_train,inputs=[is_finetune, official_dataset],outputs=[],
                            trigger_mode='once', show_progress='full')
            login_but.click(fn=logger_login,inputs=[use_logger,logger],outputs=[])
            custom_pretrained.change(fn=file_check,inputs=[custom_pretrained],outputs=[])
            self.demo = demo
            
    def get_interface(self):
        return self.demo

if __name__== "__main__" :
    demo = TrainInterface().get_interface()
    demo.queue().launch()