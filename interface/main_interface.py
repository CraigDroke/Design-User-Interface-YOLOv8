import gradio as gr
from interface.interface_methods import interface_detect

def build_interface():
    # Gradio Interface Code
    with gr.Blocks(title="YOLOv8 Interface",theme=gr.themes.Base()) as demo:
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

        update_list = [input_im,output_box_im,input_vid,output_box_vid]
        input_media = input_im
        output_media = output_box_im

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
        start_but.click(interface_detect, inputs=[input_media], outputs=output_media)
        # When these settings are changed, the change_file_type method is called
        file_type.input(change_input_type, show_progress=True, inputs=[file_type, input_media], outputs=update_list)
        
    return demo