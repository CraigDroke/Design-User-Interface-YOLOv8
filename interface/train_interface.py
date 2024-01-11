import gradio as gr

def build_train_interface():
    with gr.Blocks(title="v8 Train Interface",theme=gr.themes.Base()) as demo:
        gr.Markdown(
        """
        # Training Interface for YOLOv8
        Train your own YOLOv8 model!
        """)
        with gr.Row() as finetune_row:
            is_finetune = gr.Checkbox(label="Finetune",info="Check this box if you want to finetune a model")
            is_offical = gr.Checkbox(label="Official",info="Check this box if you want to train an official model",visible=True,interactive=True,value=True)
            custom_pretrained = gr.File(label="Pretrained Model Weights",file_count='single',type='binary',
                                    file_types=['.pt'],visible=True,show_label=True,interactive=True)
            offical_pretrained = gr.Dropdown(label="Pretrained Model",choices=["yolov8n.pt"],visible=True,interactive=True)
        # Row for start & clear buttons
        with gr.Row() as buttons:
            start_but = gr.Button(value="Start")
            
    return demo

if __name__== "__main__" :
    demo = build_train_interface()