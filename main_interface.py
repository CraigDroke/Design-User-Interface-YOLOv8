# Abstracted file for running the interface. Can run from command line or through debugger.

from interface.detect_interface import build_detect_interface
from interface.train_interface import build_train_interface
import gradio as gr

def run_main_interface():
    main = build_detect_interface()
    train = build_train_interface()
    
    return gr.TabbedInterface(interface_list=[main, train], 
                              tab_names=["Main", "Train"],
                              title="YOLOv8 Interface",
                              theme=gr.themes.Base(),
                              analytics_enabled=True)

if __name__== "__main__" :
    # run_main_interface()
    demo = run_main_interface()
    demo.queue().launch()