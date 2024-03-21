import gradio as gr
from interface.detect_interface_methods import interface_detect
from interface.defaults import shared_theme
#import skvideo.io
import numpy as np
from ffmpeg import FFmpeg
from time import sleep

class_choices = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']




class DetectInterface():
    def __init__(self):
        self.demo = None
        self.input_media = None
        self.output_media = None
        self.detect_inputs = None

    def build_detect_interface(self):
        # Gradio Interface Code
        with gr.Blocks(theme=shared_theme) as demo:
            gr.Markdown(
            """
            # Image & Video Interface for YOLOv8
            Upload your own image or video and watch YOLOv8 try to guess what it is!
            """)
            # Row for for input & output settings
            with gr.Row() as file_settings:
                # Allows choice for uploading image or video [for all]
                file_type = gr.Radio(label="File Type",info="Choose 'Image' if you are uploading an image, Choose 'Video' if you are uploading a video",
                                    choices=['Image','Video'],value='Image',show_label=True,interactive=True,visible=True)
            # Row for all inputs & outputs
            with gr.Row() as inputs_outputs:
                # Default input image: Visible, Upload from computer
                input_im = gr.Image(sources=['upload','webcam','clipboard'],type='numpy',label="Input Image",
                                    show_download_button=True,show_share_button=True,interactive=True,visible=True)
                # Default Boxed output image: Visible
                output_box_im = gr.Image(type='numpy',label="Output Image",
                                    show_download_button=True,show_share_button=True,interactive=False,visible=True)
                # Default input video: Not visible, Upload from computer
                input_vid = gr.Video(sources=['upload','webcam'],label="Input Video",
                                    show_share_button=True,interactive=True,visible=False)

                # Default Boxed output video: Not visible
                output_box_vid = gr.Video(label="Output Video",show_share_button=True,visible=False)
                show_predictions = gr.Textbox(label = 'Top Object Predictions:',visible = True, interactive= False)
                attr_box = gr.Image(type = 'numpy',label = "Attribution Map",visible = False, interactive = False)
            # List of components for clearing
            clear_list = [input_im,output_box_im,input_vid,output_box_vid,show_predictions]
            
            # Row for start & clear buttons
            with gr.Row() as buttons:
                start_but = gr.Button(value="Start")
                clear_but = gr.ClearButton(value='Clear All',components=clear_list,
                        interactive=True,visible=True)
            
            # Settings for model 
            with gr.Accordion("Model Options") as modparam_accordion:
                
                # weights options, classification threshold, bounding box checkbox, clases
                with gr.Accordion("Beginner", open=False) as modparam_accordion:
                    get_weights = gr.Radio(label="Weight Selection",info="Choose model version to use for classification",
                                        choices=['yolov8n','yolov8s','yolov8m','yolov8l','yolov8x'],value='yolov8n',show_label=True,interactive=True,visible=True,container=True)
                    get_threshold = gr.Slider(label="Classification Threshold",info="Slide to the desired threshold. This value will set the minimum confidence percentage for a object to be predicted",value = 50,minimum=0,maximum=100,step=1,show_label=True,interactive=True,visible=True,container=True)
                    with gr.Column():
                        get_class_name = gr.Dropdown(value = None, choices = class_choices, type = 'index',info= "Select classes to be shown. No selection means all classes are shown", multiselect=True, label = "Class Filter", show_label=True,interactive=True,)
                        get_boundingbox = gr.Checkbox(label= "Bounding Box Hidden", info = "Check this box if you do not want the bounding boxes to show. The top predictions will still be displayed", show_label= True, interactive= True, visible= True)
                
                # IOU slider, max detections, image size
                with gr.Accordion("Advanced", open=False) as modparam_accordion:
                    get_iou = gr.Slider(label="IOU Threshold",info="Slide to the desired threshold. This measures the overlap between bounding boxes, the lower the value the more detections",value = 50,minimum=0,maximum=100,step=1,show_label=True,interactive=True,visible=True,container=True)
                    get_max_det = gr.Slider(label="Maximum Detections",info="Slide to the desired number. This value sets the maximum number of bounding boxes allowed",value = 300,minimum=0,maximum=1000,step=10,show_label=True,interactive=True,visible=True,container=True)
                    get_size = gr.Slider(label="Image SIze",info="Slide to the desired image size. Must be between 32 and 4096",value = 640,minimum=32,maximum=4096,step=32,show_label=True,interactive=True,visible=True,container=True)
                
                # visualize checkbox, agnostic checkbox, pretrained file
                with gr.Accordion("Expert", open=False) as modparam_accordion:
                    pretrained_file = gr.File(file_count='single',file_types=['.pt'],label='Pretrained Model Weights',type='filepath',show_label=True,container=True,interactive=True,visible=True)
                    get_visualize = gr.Checkbox(label = "Visualize Model Features", info = "Shows the features of the image the model uses for classification", show_label= True, interactive = True, visible = True)
                    get_agnostic = gr.Checkbox(label= "Class Agnostic NMS",info = "Will set a bouning box around all objects, including unknown items", show_label = True, interactive = True, visible = True)


                update_list = [input_im,output_box_im,input_vid,output_box_vid,show_predictions,attr_box]
                self.input_media = input_im 
                self.output_media = [output_box_im,output_box_vid,show_predictions,attr_box]
                self.detect_inputs = [input_im, input_vid, get_weights,get_threshold,pretrained_file,get_iou,get_max_det, get_agnostic,get_size,get_visualize,get_class_name, get_boundingbox]
                
            def change_input_type(file_type):
                if file_type == 'Image':
                    # self.input_media = input_im
                    # self.detect_inputs = [self.input_media, get_weights, get_threshold, pretrained_file, get_iou, get_max_det, get_agnostic, get_size, get_visualize, get_class_name, get_boundingbox]
                    # self.output_media = [output_box_im, show_predictions]
                    # self.change_input_type(input_im, output_box_im)
                    return {
                        input_im: gr.Image(visible=True),
                        output_box_im: gr.Image(visible=True),
                        input_vid: gr.Video(visible=False),  # Ensure input_vid remains Video type
                        output_box_vid: gr.Video(visible=False),
                        show_predictions: gr.Textbox(visible=True),
                        attr_box: gr.Image(visible=False)
                    }
                elif file_type == 'Video':
                    # Define a function to update detect_inputs
                    # self.input_media = input_vid
                    # self.detect_inputs = [self.input_media, get_weights, get_threshold, pretrained_file, get_iou, get_max_det, get_agnostic, get_size, get_visualize, get_class_name, get_boundingbox]
                    # self.output_media = [output_box_vid, show_predictions]
                    # self.change_input_type(input_vid, output_box_vid)
                    return {
                        input_im: gr.Image(visible=False),
                        output_box_im: gr.Image(visible=False),
                        input_vid: gr.Video(visible=True),  # Ensure input_vid remains Video type
                        output_box_vid: gr.Video(visible=True),
                        show_predictions: gr.Textbox(visible=False),
                        attr_box: gr.Image(visible=False)
                    }
                
            def change_viz(get_visualize):
                if get_visualize:
                    return {
                        show_predictions: gr.Textbox(visible=False),
                         
                    }
                else:
                    return {
                        show_predictions: gr.Textbox(visible=True)
                    }
            def set_attr():
                return {
                    input_im: gr.Image(visible=False),
                    output_box_im: gr.Image(visible=True),
                    input_vid: gr.Video(visible=False),  # Ensure input_vid remains Video type
                    output_box_vid: gr.Video(visible=False),
                    show_predictions: gr.Textbox(visible=True),
                    attr_box: gr.Image(visible=True)
                }


            # When start button is clicked, the run_all method is called
            start_but.click(interface_detect, inputs=self.detect_inputs, outputs=self.output_media)
            # When these settings are changed, the change_file_type method is called
            file_type.input(change_input_type, show_progress=True, inputs=[file_type], outputs=update_list)
            get_visualize.input(change_viz,inputs=get_visualize,outputs=show_predictions)
            demo.load(change_input_type, show_progress=True, inputs=[file_type], outputs=update_list)
            self.demo = demo
            start_but.click(fn=set_attr,inputs = [],outputs = update_list)


    def change_input_type(self, input, output):
        self.input_media = input
        self.detect_inputs[0] = input
        self.output_media[0] = output

    def get_interface(self):
        self.build_detect_interface()
        return self.demo
    
    