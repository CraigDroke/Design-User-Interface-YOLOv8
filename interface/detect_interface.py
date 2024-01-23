import gradio as gr
from interface.detect_interface_methods import interface_detect
from interface.defaults import shared_theme

class_choices = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def build_detect_interface():
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
        
        # List of components for clearing
        clear_list = [input_im,output_box_im,input_vid,output_box_vid]
        
        # Row for start & clear buttons
        with gr.Row() as buttons:
            start_but = gr.Button(value="Start")
            clear_but = gr.ClearButton(value='Clear All',components=clear_list,
                    interactive=True,visible=True)
        
        with gr.Accordion("Model Options") as modparam_accordion:
            with gr.Accordion("Beginner") as modparam_accordion:
                get_weights = gr.Radio(label="Weight Selection",info="Choose weights for model to use for classification",
                                    choices=['yolov8n','yolov8s','yolov8m','yolov8l','yolov8x'],value='yolov8n',show_label=True,interactive=True,visible=True,container=True)
                get_threshold = gr.Slider(label="Classification Threshold",info="Slide to the desired threshold",value = 50,minimum=0,maximum=100,step=1,show_label=True,interactive=True,visible=True,container=True)
                with gr.Column():
                    get_class_name = gr.Dropdown(value = None, choices = class_choices, type = 'index', multiselect=True, label = "Class Filter", show_label=True,interactive=True,)
            with gr.Accordion("Advanced") as modparam_accordion:
                get_iou = gr.Slider(label="IOU Threshold",info="Slide to the desired threshold",value = 50,minimum=0,maximum=100,step=1,show_label=True,interactive=True,visible=True,container=True)
                get_max_det = gr.Slider(label="Maximum Detections",info="Slide to the desired number",value = 300,minimum=0,maximum=1000,step=10,show_label=True,interactive=True,visible=True,container=True)
                get_size = gr.Slider(label="Image SIze",info="Slide to the desired image size",value = 640,minimum=32,maximum=4096,step=32,show_label=True,interactive=True,visible=True,container=True)
            with gr.Accordion("Expert") as modparam_accordion:
                pretrained_file = gr.File(file_count='single',file_types=['.pt'],label='Pretrained Model Weights',type='filepath',show_label=True,container=True,interactive=True,visible=True)
                get_visualize = gr.Checkbox(label = "Visualize Model Figures", show_label= True, interactive = True, visible = True)
                get_agnostic = gr.Checkbox(label= "Class Agnostic NMS",info = "Will box all objects, including unknown items", show_label = True, interactive = True, visible = True)

        update_list = [input_im,output_box_im,input_vid,output_box_vid]
        input_media = input_im 
        output_media = output_box_im
        detect_inputs = [input_media,get_weights,get_threshold,pretrained_file,get_iou,get_max_det, get_agnostic,get_size,get_visualize,get_class_name]

        def change_input_type(file_type, input_media):
            if file_type == 'Image':
                input_media = input_im
                output_media = output_box_im
                return {
                    input_im: gr.Image(visible=True),
                    output_box_im: gr.Image(visible=True),
                    input_vid: gr.Video(visible=False),
                    output_box_vid: gr.Video(visible=False)
                }
            elif file_type == 'Video':
                input_media = input_vid
                output_media = output_box_vid
                return {
                    input_im: gr.Image(visible=False),
                    output_box_im: gr.Image(visible=False),
                    input_vid: gr.Video(visible=True),
                    output_box_vid: gr.Video(visible=True)
                }
        

        # When start button is clicked, the run_all method is called
        start_but.click(interface_detect, inputs=detect_inputs, outputs=output_media)
        # When these settings are changed, the change_file_type method is called
        file_type.input(change_input_type, show_progress=True, inputs=[file_type, input_media], outputs=update_list)
        
    return demo