import gradio as gr

def build_train_interface():
    with gr.Blocks(title="v8 Train Interface",theme=gr.themes.Base()) as demo:
        gr.Markdown(
        """
        # Training Interface for YOLOv8
        Train your own YOLOv8 model!
        """)
        
    return demo