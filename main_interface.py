# Abstracted file for running the interface. Can run from command line or through debugger.

from interface.detect_interface import build_detect_interface
from interface.train_interface import TrainInterface
from interface.resources_interface import build_resources_interface
import gradio as gr
from interface.defaults import shared_theme
import wandb

def build_main_interface():
    detect = build_detect_interface()
    train = TrainInterface().get_interface()
    resources = build_resources_interface()
    
    with gr.Blocks(title="YOLOv8 Interface",theme=shared_theme) as demo:
        gr.Markdown(
        """
        # YOLOv8 Interface
        Choose between the Detect and Train interfaces.
        """)
        gr.TabbedInterface(interface_list=[detect, train, resources], 
                            tab_names=["Detect", "Train", "Resources"],
                            theme=shared_theme,
                            analytics_enabled=True)
        
    return demo

if __name__== "__main__" :
    # run_main_interface()
    demo = build_main_interface()
    demo.queue().launch(inbrowser=True,
                        share=False,
                        debug=False,
                        show_error=False,
                        quiet=False)
    demo.integrate(wandb=wandb)