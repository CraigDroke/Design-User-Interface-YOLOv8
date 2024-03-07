import gradio as gr
from interface.defaults import shared_theme
from interface.train_interface_methods import interface_train, interface_login, interface_train_tensorboard

pretrained = ""
dataset = ""
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
            epochs = gr.Slider(label="Epochs",minimum=1,maximum=10,step=1,value=2,visible=True,interactive=True)
            custom_pretrained = gr.File(label="Pretrained Model Weights",file_count='single',type='binary',
                                    file_types=['.pt'],visible=True,show_label=True,interactive=True)
            official_pretrained = gr.Dropdown(label="Pretrained Model",choices=["yolov8n.pt"],visible=True,interactive=True)
            
        with gr.Row() as dataset_row:
            is_official_dataset = gr.Checkbox(label="Official",info="Check this box if you want to use an official dataset",visible=True,interactive=True,value=True)
            custom_dataset = gr.File(label="Custom Dataset",file_count='single',type='binary',
                                    file_types=['.zip'],visible=True,show_label=True,interactive=True)
            official_dataset = gr.Dropdown(label="Dataset",choices=["coco128"],visible=True,interactive=True)

        # Row for start & clear buttons
        with gr.Row() as buttons:
            start_but = gr.Button(value="Start")
        with gr.Accordion("Logger Options") as login_accordion:
            use_logger = gr.Checkbox(label="Use Logger",info="Check this box if you want to use a logger",visible=True,interactive=True,value=True)
            logger = gr.Radio(choices=['WANDB', 'Tensorboard'],value='WANDB',show_label=True,interactive=True,visible=True,
                              label="Logger",info="Choose which logger to use")
            login_but = gr.Button(value="Login")
            key = gr.Textbox(label="API Key",placeholder="Enter your API key",visible=True,interactive=True)
        
        def update_pretrained(custom_pretrained,official_pretrained):
            if custom_pretrained is not None:
                pretrained = custom_pretrained
            else:
                pretrained = official_pretrained

            print(f"Model:{pretrained}")


        def update_dataset(custom_dataset,official_dataset):
            if custom_dataset is not None:
                dataset = custom_dataset
            else:
                dataset = official_dataset
        


        start_but.click(fn=interface_train,inputs=[is_finetune,epochs],outputs=[])
        #login_but.click(fn=interface_login,inputs=[logger],outputs=[])
        login_but.click(fn=interface_login,inputs=[logger,epochs,key],outputs=[])

        
        custom_pretrained.upload(fn = update_pretrained,inputs = [custom_pretrained,official_pretrained],outputs = [])
        custom_dataset.upload(fn = update_dataset,inputs = [custom_dataset,official_dataset],outputs = [])
        official_pretrained.change(fn = update_pretrained,inputs = [custom_pretrained,official_pretrained],outputs = [])
        official_dataset.change(fn = update_dataset,inputs = [custom_dataset,official_dataset],outputs = [])
    return demo

if __name__== "__main__" :
    demo = build_train_interface()