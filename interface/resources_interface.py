import gradio as gr
from interface.defaults import shared_theme

def build_resources_interface():
    # Gradio Interface Code
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
        """
        # Interface Wiki


        ## How to Run the Interface
        1. First download the requirements using 'pip install -r requirements.txt' in your environment (This may be a time consuming process).
        2. Next, open the main_interface.py file in your prefered IDE.
        3. Run and Debug the code. 
        4. Open a terminal in the IDE and look for the URL output. Ctr click the link to open in Gradio Interface in your default browser.

        ## Using the Detect Interface
        1. Start by uploading a picture using any of the 3 inputs. These are webcam (device camera), image import, and drag and drop functionality. 
        2. Once the image is uploaded, you can adjust settings such as the confidence threshold and whether or not to use non-maximum suppression.
        3. Once satisfied with the settings, click the "Start" button to see the results of the YOLOv8 model.
        4. This process can be repeated for videos as well.

        ## Using the Train Interface
        1. To use the training interface, start by determining if you want to use an offical or custom model and how many epochs you want to train for.
        2. Next, either select an offical model from the list, or upload your own weights.
        3. Then, either select an offical dataset from the list, or upload your own dataset.
        4. If you want to use a logger, select the logger you want to use.
        5. Click the "Start" button to begin, or the "Login" button if using one to start training.

        ### Understanding the Repository (Deep-Dive into the File Structure)
        
        | Folder: interface       | Summary                                                                       |
        |-------------------------|-------------------------------------------------------------------------------|
        | bio_interface           | This file contains the biographies of all students who worked on the project  |
        | detect_interface_methods| The methods that help power the YOLO detection input/output                   |
        | detect_interface        | This file contains the gradio code to power our interface                     |
        | resources_interface     | The current page you're reading                                               |
        | train_interface_methods | The methods that power the YOLO training algorithms                           |
        | train_interface         | This file contains the gradio code to power our interface                     |
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

