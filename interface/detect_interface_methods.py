import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
import os, os.path
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',14: 'bird', 15: 'cat',16: 'dog',17: 'horse',18: 'sheep',19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',69: 'oven',70: 'toaster',71: 'sink',72: 'refrigerator',73: 'book',74: 'clock',75: 'vase',76: 'scissors',77: 'teddy bear',78: 'hair drier',79: 'toothbrush'}
dirname = os.path.dirname(__file__)

dirname = dirname.split("Clinic_2")[0] + "Clinic_2"
#print (dirname)
def count_train_folders(directory):
    count = 0
    for folder in os.listdir(directory):
        if "predict" in folder:
            count += 1
    return count

run = count_train_folders(os.path.join(dirname, 'runs', 'detect'))


def park_demo():
    return interface_detect(asarray(Image.open('C:\\Users\\modon\\Documents\\Clinic_2\\park.jpeg')),None,None,13,"yolov8x.pt",50,300,False,640,False,[],False)

def cars_demo():
    return interface_detect(asarray(Image.open('C:\\Users\\modon\\Documents\\Clinic_2\\cars.jpg')),None,None,20,"yolov8l.pt",50,300,False,640,False,[],True)

def video_demo():
    return interface_detect(None,'C:\\Users\\modon\\Documents\\Clinic_2\\ducks.mp4',None,50,"yolov8x.pt",50,300,False,640,False,[],True)

def interface_detect(source_im,source_vid,weights,thres,pretrained,user_iou,user_det, get_agnostic,img_size,viz,get_class_name, get_boundingbox):
    
    #print("source: ", source)
    global run, output_string
    class_ids = []
    precentages = []
    classes = []
    output_string = ""
    if source_im is not None:
        source = source_im
    elif source_vid is not None:
        source = source_vid
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
        results = model.predict(source=source,conf=thres/100, iou = user_iou/100,max_det = user_det, agnostic_nms = get_agnostic,imgsz = img_size,visualize = viz,classes = get_class_name,boxes=False)
        if viz:
            run = run + 1
            
            
            return [os.path.join(dirname,"runs","detect","predict")+ str(run) +"\\image0\\stage0_Conv_features.png",os.path.join(dirname,"test.mp4"),""]
        else:
            for result in results:
                for box in result.boxes:
                    class_ids.append(int(box.cls))
                    precentages.append(str(round(box.conf.item(), 2)))
                if get_boundingbox:
                    result.boxes = []

            for i in range(len(class_ids)):
                classes.append(class_names.get(class_ids[i]))
                output_string += "Class name: " + classes[i] + " Confidence: " + precentages[i] + '\n'



        return [results[0].plot(), None, output_string]
            
        
    
    elif source is not None:
        run2 = random.randint(0,100000)
        cap = cv2.VideoCapture(source)
        # Get the properties of the input video
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "Tracked_"+str(run2)+".mp4"
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

        
        return [None, output_path,""]
    
    
    







    #else:
        #raise ValueError("Invalid source type")
    

    
run = run