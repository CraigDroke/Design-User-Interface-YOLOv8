# Abstracted file for running the interface. Can run from command line or through debugger.

from interface.detect_interface import DetectInterface
from interface.train_interface import build_train_interface
from interface.resources_interface import build_resources_interface
from interface.bio_interface import build_bio_interface 
import gradio as gr
from interface.defaults import shared_theme

def build_main_interface():
    detect = DetectInterface().get_interface()
    train = build_train_interface()
    resources = build_resources_interface()
    bio = build_bio_interface()
    
    with gr.Blocks(title="YOLOv8 Interface",theme=shared_theme) as demo:
        gr.Markdown(
        """
        # YOLOv8 Interface
        Choose between the Detect and Train interfaces.
        """)
        gr.TabbedInterface(interface_list=[detect, train, resources, bio], 
                            tab_names=["Detect", "Train", "Resources", "Bio"],
                            theme=shared_theme,
                            analytics_enabled=True)
        
    return demo

if __name__== "__main__" :
    # run_main_interface()
    demo = build_main_interface()
    demo.queue().launch(debug=True)