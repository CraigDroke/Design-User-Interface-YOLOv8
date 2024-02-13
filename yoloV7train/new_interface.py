import gradio as gr
import argparse
import sys
import os
from PIL import Image
sys.path.append('Interface_Dependencies')
#sys.path.append('Interface')
sys.path.append('yoloV7train')
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import numpy as np

shared_theme = gr.themes.Base()

#from train import data

from Interface.detect_interface import build_detect_interface
from Interface.train_interface import build_train_interface

import gradio as gr


def build_main_interface():
    detect = build_detect_interface()
    train = build_train_interface()
    
    with gr.Blocks(title="YOLOv7 Interface",theme=shared_theme) as demo:
        gr.Markdown(
        """
        # YOLOv7 Interface
        Choose between the Detect and Train interfaces.
        """)
        gr.TabbedInterface(interface_list=[detect, train], 
                            tab_names=["Detect", "Train"],
                            theme=shared_theme,
                            analytics_enabled=True)
    
            
    return demo

if __name__== "__main__" :
    # run_main_interface()
    demo = build_main_interface()
    demo.queue().launch()