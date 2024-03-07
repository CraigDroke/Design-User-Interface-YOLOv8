import gradio as gr
from interface.defaults import shared_theme

def build_resources_interface():
    # Gradio Interface Code
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
        """
        # Interface Wiki


        ## How to Run the Interface
        1. First download the requirements file using 'pip install -r requirements.txt' in your environment (This may be a time consuming process).
        2. Next, open the main_interface.py file in your prefered IDE.
        3. Run and Debug the code. 
        4. Open a terminal in the IDE and look for the URL output. Ctr click the link to open in Gradio Interface in your default browser.

        ## Using the Interface (Features)
        1. Start by uploading a picture using any of the 3 features. These is webcam (device camera), image import, and drag and drop functionality. 
        TODO - Finish this section

        ### Understanding the Repository (Deep-Dive into the File Structure)
        | Folder: interface       | Summary                                                                       |
        |-------------------------|-------------------------------------------------------------------------------|
        | bio_interface           | This file contains the bio stuff... yanno                                     |
        | detect_interface_methods| The methods that help power the YOLO detection input/output                   |
        | detect_interface        | This detects the gismos and do dads                                           |
        | resources_interface     | The current page you're reading                                               |
        | train_interface_methods | The methods that power the YOLO training algorithms                           |
        | train_interface         | This trains the interface to detect the gismos and the trinkets               |
        | main_interface          | This is the main interface file that abstracts the complex functions          |                                                                             |  
        | .gitignore              | Forces git to ignore unnecessary or overly large files                        |

        ## YOLOv8 Code & Documentation
        YOLOv8 is made by a company called Ultralytics. Most important info is on their Github, website, and youtube channel.
        1. [Ultralytics (YOLOv8) Github](https://github.com/ultralytics/ultralytics)
        2. [YOLOv8 Documentation Website](https://docs.ultralytics.com/)
        3. [Ultralytics Youtube Channel](https://www.youtube.com/ultralytics)
        4. [Join the Ultralytics Discord](https://discord.com/invite/ultralytics-1089800235347353640)
        """)
    return demo

