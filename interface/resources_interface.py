import gradio as gr
from interface.defaults import shared_theme

def build_resources_interface():
    # Gradio Interface Code
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
        """
        # How to Use:

        ### In order to successfully run the interface, the first step would be to choose the file you are planning on inputting, either a image or a video. 
        ### Before you click run, you have the options to change many of the settings that will affect the output results.
        ### The settings are seperating in different knowledge levels. It is split between 'beginner', 'advanced', and lastly 'expert'. 
        ### Once the settings are set up to your liking and the file is inputted into the interface, you can click run and it will display the results.
        ### When you are done with that file upload and their results, click clear and it will clear the data on the interface. 

        # Settings Overveiw: 

        # Output Explanation:

        # Helpful Resources for YOLOv8
        This page has a list of websites & guides to help you become a YOLOv8 Pro!
        ## YOLOv8 Code & Documentation
        YOLOv8 is made by a company called Ultralytics. Most important info is on their Github, website, and youtube channel.
        1. [Ultralytics (YOLOv8) Github](https://github.com/ultralytics/ultralytics)
        2. [YOLOv8 Documentation Website](https://docs.ultralytics.com/)
        3. [Ultralytics Youtube Channel](https://www.youtube.com/ultralytics)
        4. Join the Ultralytics Discord server as well!
        """)
    return demo