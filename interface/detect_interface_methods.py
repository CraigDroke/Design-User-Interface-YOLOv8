import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
import os, os.path

lst = os.listdir('C:\\Users\\modon\\Documents\\Clinic_2\\runs\\detect')
run = len(lst)
def interface_detect(source,weights,thres,pretrained,user_iou,user_det, get_agnostic,img_size,viz,get_class_name):
    global run
    # Load a pretrained YOLOv8n model
    #print(pretrained)
    if pretrained is not None:
        model = YOLO(pretrained)
    else:
        if not weights.endswith(".pt"):
            weights = weights + ".pt"
        model = YOLO(weights)  # Load an official Detect model
    if isinstance(source, np.ndarray):
        if get_class_name == []:
            get_class_name = None
        results = model.predict(source=source,conf=thres/100, iou = user_iou/100,max_det = user_det, agnostic_nms = get_agnostic,imgsz = img_size,visualize = viz,classes = get_class_name)
        if viz:
            run = run + 1
            
            return "C:\\Users\\modon\\Documents\\Clinic_2\\runs\\detect\\predict"+ str(run) +"\\image0\\stage0_Conv_features.png"
             
        else:
            print(results)
            return results[0].plot()
    elif source.endswith(".mp4"):
        cap = cv2.VideoCapture(source)
        # Get the properties of the input video
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "Tracked_" + source
        # Output video writer. Must have same frame rate, height, and width as the input video
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), input_fps, (input_width, input_height))
    
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 inference on the frame
                results = model(frame)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                output.write(annotated_frame)

            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object, writer object, and close the display window
        output.release()
        cap.release()
        cv2.destroyAllWindows()
        return output_path
    else:
        raise ValueError("Invalid source type")
    
run = run
