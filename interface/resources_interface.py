import gradio as gr
from interface.defaults import shared_theme

def build_resources_interface():
    # Gradio Interface Code
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
        """
        # Helpful Resources for YOLOv8
        This page has a list of websites & guides to help you become a YOLOv8 Pro!
        ## YOLOv8 Code & Documentation
        YOLOv8 is made by a company called Ultralytics. Most important info is on their Github, website, and youtube channel.
        1. [Ultralytics (YOLOv8) Github](https://github.com/ultralytics/ultralytics)
        2. [YOLOv8 Documentation Website](https://docs.ultralytics.com/)
        3. [Ultralytics Youtube Channel](https://www.youtube.com/ultralytics)
        3. Join the Ultralytics Discord server as well!
        """)
    return demo