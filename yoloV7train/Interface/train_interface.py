import gradio as gr
import sys

shared_theme = gr.themes.Base()
sys.path.append('Interface_Dependencies')
sys.path.append('Interface')
sys.path.append('yoloV7train')
sys.path.append('./') 

from train_interface_metds import interface_train, interface_login
from Interface_Dependencies.run_methods import run_all

def build_train_interface():
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
        """
        # Training Interface for YOLOv7
        Train your own YOLOv7 model!
        """) 

        with gr.Row() as file_settings:
                # Allows choice for uploading image or video [for all]
                file_type = gr.Radio(label="File Type",info="Choose 'Default' if you are using default values uploaded by server, Choose 'Upload' if you are uploading your own files from your computer",
                                choices=['Default','Upload'],value='Default',show_label=True,interactive=True,visible=True)
                # Allows choice of source, from computer or webcam [for all]
                source_type = gr.Radio(label="Source Type",info="Choose 'Computer' if you are uploading from your computer",
                                choices=['Computer'],value='Computer',show_label=True,interactive=True,visible=True)
                # Allows choice of which smooth gradient output to show (1-3) [only for images]
                #output_map = gr.Slider(label="Map Output Number",info="Choose a whole number from 1 to 3 to see the corresponding attribution map",
                                #minimum=1,maximum=3,value=1,interactive=True,step=1,show_label=True)
                # dev_type = gr.Radio(label="Device Type",info="Choose 'CPU' if you have no gpu, Choose 'Cuda' if you have a cuda enabled gpu",
                #                 choices=['CPU','Cuda'],value='CPU',show_label=True,interactive=True,visible=True)
                 
                 
        with gr.Row() as model_settings:
                inf_size = gr.Number(label='Inference Size (pixels)',value=640,precision=0, interactive=True)
                pgt_lr = gr.Number(label='Learning rate for plausibility gradient',value=0.0, interactive=True)
                pgt_lr_decay = gr.Number(label='decay learning rate for plausibility gradient',value=0.75, interactive=True)
                agnostic_nms = gr.Checkbox(label='Agnostic NMS',value=True, interactive=True)
                pgt_lr_decay_step = gr.Number(label='Step for decal learning rate',value=20.0, interactive=True)
                epochs = gr.Number(label='# of training loops',value=0.45, interactive=True)
                conf_thres = gr.Number(label='object confidence threshold',value=0.50, interactive=True)
                batch_size = gr.Number(label='total batch size for GPU',value=10, interactive=True)
                device = gr.Slider(label='cuda device, i.e. 0 or 0,1,2,3', minimum=0, maximum=3, value=1, step=1, visible=True, interactive=True)
                default_device = gr.Checkbox(label='cpu', info = 'Check if you have no cuda devices', visible=True,interactive=True)
                # Weights File Upload
                #weights = gr.File(label='Weights File',type='file',file_count='single',file_types=["pt"],value="weights/yolov7.pt")  
                   
        
        with gr.Row() as inputs_outputs:
            # Default input image: Visible, Upload from computer
            is_finetune = gr.Checkbox(label="Finetune",info="Check this box if you want to finetune a model")

            # Default Boxed output image: Visible
            is_offical_pretrained = gr.Checkbox(label="Official",info="Check this box if you want to train an official model",visible=True,interactive=True,value=True, show_label=True)

            custom_pretrained = gr.File(label='Weights File',file_count='single', type='binary',file_types=["pt"], visible=True,show_label=True,interactive=True)  
            
            offical_pretrained = gr.Dropdown(label="Pretrained Model",file_count='single', type='index', choices=["yolov7.pt"], value="weights/yolov7.pt", visible=True,interactive=True, show_label=True, show_download_button=True)

            is_official_dataset = gr.Checkbox(label="Official",info="Check this box if you want to use an official dataset",visible=True,interactive=True,value=True,show_label=True)
            
            custom_dataset = gr.File(label="Custom Dataset",file_count='single',type='binary',
                                        file_types=['.zip'],visible=True,show_label=True,interactive=True)
            
            official_dataset = gr.Dropdown(label="Dataset",choices=["coco128"], value="yoloV7train/coco128.zip", show_download_button = True, visible=True,interactive=True,show_label=True)
            
            custom_data_file = gr.File(label="Custom Data file",file_count='single',type='binary',
                                        file_types=['.yaml'],visible=True,show_label=True,interactive=True)
            
            official_data_file = gr.Dropdown(label="Data file",choices=["coco128.yaml"], value="data/coco128.yaml", visible=True,interactive=True,show_label=True)
            
            custom_hyp_file = gr.File(label="Custom hyp file",file_count='single',type='binary',
                                        file_types=['.yaml'],visible=True,show_label=True,interactive=True)
            
            official_hyp_file = gr.Dropdown(label="hyp file",choices=["hyp.scratch.p6.yaml"], value="data/hyp.scratch.p6.yaml", visible=True,interactive=True,show_label=True)

            formatted_time = gr.Textbox(label = 'Time to Run in Seconds:', value = "", show_label=True)

            output_project = gr.File(type='file',label="Output project file",
                             show_download_button=True,show_share_button=True,interactive=False,visible=True)
            
            output_weight_file = gr.File(type='file',label="Output weight file",
                             show_download_button=True,show_share_button=True,interactive=False,visible=True)

       
        clear_comp_list = [output_project, output_weight_file, formatted_time, is_finetune, is_offical_pretrained, 
                           custom_pretrained, offical_pretrained, is_official_dataset, custom_dataset, official_dataset, 
                           custom_data_file, official_data_file, custom_hyp_file, official_hyp_file, device, default_device]
        
       
        # Row for start & clear buttons

        with gr.Row() as buttons:
                start_but = gr.Button(label="Start")
                clear_but = gr.ClearButton(value='Clear All',components=clear_comp_list,
                   interactive=True,visible=True)
        
        with gr.Accordion("Logger Options") as login_accordion:
                use_logger = gr.Checkbox(label="Use Logger",info="Check this box if you want to use a logger",visible=True,interactive=True,value=True)
                logger = gr.Radio(choices=['WANDB', 'ClearML', 'Tensorboard'],value='WANDB',show_label=True,interactive=True,visible=True,
                                label="Logger",info="Choose which logger to use")
                login_but = gr.Button(value="Login")
            
        start_but.click(fn=interface_train,inputs=[device, is_finetune, is_offical_pretrained, custom_pretrained, offical_pretrained, is_official_dataset, custom_dataset, official_dataset, custom_data_file, official_data_file, custom_hyp_file, official_hyp_file],outputs=[])
        clear_but.click(fn=interface_train,inputs=[is_finetune, is_offical_pretrained, custom_pretrained, offical_pretrained, is_official_dataset, custom_dataset, official_dataset, custom_data_file, official_data_file, custom_hyp_file, official_hyp_file],outputs=[])
        login_but.click(fn=interface_login,inputs=[logger],outputs=[])


        def change_file_type(file, source):
                """
                Changes the visible components of the gradio interface

                Args:
                file (str): Type of the file (image or video)
                source (str): If the file is uploaded or from webcam
                is_stream (bool): If the video is streaming or uploaded
                Returns:
                Dictionary: Each component of the interface that needs to be updated.
                """
                if file == "Default":
                                return {
                                output_project: gr.File(visible=True), 
                                output_weight_file: gr.File(visible=True), 
                                formatted_time: gr.Textbox(visible=True), 
                                is_finetune: gr.File(visible=True), 
                                is_offical_pretrained: gr.Checkbox(visible=True), 
                                custom_pretrained: gr.File(label='Weights File',file_count='single', type='binary',file_types=["pt"], visible=False,show_label=True,
                                        show_download_button=True,show_share_button=True,interactive=True), 
                                offical_pretrained: gr.Dropdown(visible=True), 
                                is_official_dataset: gr.Checkbox(visible=True), 
                                custom_dataset: gr.File(label="Custom Dataset",file_count='single',type='binary',
                                                        file_types=['.zip'],show_download_button=True,show_share_button=True, visible=False,show_label=True,interactive=True),
                                official_dataset: gr.Dropdown(visible=True), 
                                custom_data_file: gr.File(label="Custom Data file",file_count='single',type='binary',
                                                        file_types=['.yaml'], show_download_button=True, show_share_button=True, visible=False, show_label=True, interactive=True), 
                                official_data_file: gr.Dropdown(visible=True), 
                                custom_hyp_file: gr.File(label="Custom hyp file",file_count='single',type='binary',
                                                        file_types=['.yaml'], show_download_button=True,show_share_button=True, visible=False, show_label=True, interactive=True), 
                                official_hyp_file: gr.Dropdown(visible=True),
                                formatted_time: gr.Textbox(visible=True),
                                default_device: gr.Checkbox(visible=True),
                                device: gr.Slider(visible=True)
                                }
                
                elif file == "Upload":
                        if source == "Computer":
                                        return {
                                        output_project: gr.File(visible=True), 
                                        output_weight_file: gr.File(visible=True), 
                                        formatted_time: gr.Textbox(visible=True), 
                                        is_finetune: gr.File(visible=True), 
                                        is_offical_pretrained: gr.Checkbox(visible=False), 
                                        custom_pretrained: gr.File(label='Weights File',file_count='single', type='binary',file_types=["pt"], visible=True,show_label=True,
                                                show_download_button=True,show_share_button=True,interactive=True), 
                                        offical_pretrained: gr.Dropdown(visible=False), 
                                        is_official_dataset: gr.Checkbox(visible=False), 
                                        custom_dataset: gr.File(label="Custom Dataset",file_count='single',type='binary',
                                                                file_types=['.zip'],show_download_button=True,show_share_button=True, visible=True,show_label=True,interactive=True),
                                        official_dataset: gr.Dropdown(visible=False), 
                                        custom_data_file: gr.File(label="Custom Data file",file_count='single',type='binary',
                                                                file_types=['.yaml'], show_download_button=True, show_share_button=True, visible=True, show_label=True, interactive=True), 
                                        official_data_file: gr.Dropdown(visible=False), 
                                        custom_hyp_file: gr.File(label="Custom hyp file",file_count='single',type='binary',
                                                                file_types=['.yaml'], show_download_button=True,show_share_button=True, visible=True, show_label=True, interactive=True), 
                                        official_hyp_file: gr.Dropdown(visible=False),
                                        formatted_time: gr.Textbox(visible=True),
                                        default_device: gr.Checkbox(visible=True),
                                        device: gr.Slider(visible=True)
                                        }
                
                        
        change_comp_list = [output_project, output_weight_file, formatted_time, is_finetune, is_offical_pretrained, 
                                custom_pretrained, offical_pretrained, is_official_dataset, custom_dataset, official_dataset, 
                                custom_data_file, official_data_file, custom_hyp_file, official_hyp_file, device, default_device]
        
        # change_comp_list = [output_project, output_weight_file, formatted_time, is_finetune, is_offical_pretrained, 
        #                         custom_pretrained, offical_pretrained, is_official_dataset, custom_dataset, official_dataset, 
        #                         custom_data_file, official_data_file, custom_hyp_file, official_hyp_file]

        # List of gradio components that are input into the run_all method (when start button is clicked)
        run_inputs = [pgt_lr, pgt_lr_decay, pgt_lr_decay_step, conf_thres, epochs, conf_thres, batch_size, default_device, device, inf_size, agnostic_nms, is_finetune, is_offical_pretrained, custom_pretrained, offical_pretrained, is_official_dataset, custom_dataset, official_dataset, custom_data_file, official_data_file, custom_hyp_file]
        # List of gradio components that are output from the run_all method (when start button is clicked)
        run_outputs = [output_project, formatted_time, output_weight_file]
        
        # When these settings are changed, the change_file_type method is called
        file_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type], outputs=change_comp_list)
        source_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type], outputs=change_comp_list)
        # When start button is clicked, the run_all method is called
        start_but.click(run_all, inputs=run_inputs, outputs=run_outputs)
        # When the demo is first started, run the change_file_type method to ensure default settings
        demo.load(change_file_type, show_progress=True, inputs=[file_type, source_type], outputs=change_comp_list)
    return demo
                
if __name__== "__main__" :
    demo = build_train_interface()