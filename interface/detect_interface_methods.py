import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
import os, os.path
import random
import torch
import matplotlib.pyplot as plt
from ultralytics.models.yolo.detect import DetectionTrainer
from PIL import Image

class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',14: 'bird', 15: 'cat',16: 'dog',17: 'horse',18: 'sheep',19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',69: 'oven',70: 'toaster',71: 'sink',72: 'refrigerator',73: 'book',74: 'clock',75: 'vase',76: 'scissors',77: 'teddy bear',78: 'hair drier',79: 'toothbrush'}
dirname = os.path.dirname(__file__)
dirname = dirname.split("Clinic_2")[0] + "Clinic_2"
#print (dirname)



def my_train(model,trainer,img):
    imgt = torch.tensor(img,requires_grad=True, dtype=torch.float64)
    model.trainer = trainer
    results = model.predict(imgt)
    arr = results[0].plot()
    for result in results:
        img_width = imgt.shape[2]
        img_height = imgt.shape[3]
        open('dataset/labels/train/your_file.txt', 'w').close()
        with open('dataset/labels/train/your_file.txt', 'a') as f:
            for box in result.boxes:
                class_id = int(box.cls)
                coords = box.xywh.cpu().tolist()
                dw = 1.0 / img_width
                dh = 1.0 / img_height
                x = (coords[0][0] + coords[0][2]) / 2.0
                y = (coords[0][1] + coords[0][3]) / 2.0
                w = coords[0][2] - coords[0][0]
                h = coords[0][3] - coords[0][1]
                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh
                yolo_format = "{} {} {} {} {}\n".format(class_id, x, y, w, h)
                f.write(yolo_format)

    for result in results:
        img_width = imgt.shape[2]
        img_height = imgt.shape[3]
        open('dataset/labels/train/your_file1.txt', 'w').close()
        with open('dataset/labels/train/your_file1.txt', 'a') as f:
            for box in result.boxes:
                class_id = int(box.cls)
                coords = box.xywh.cpu().tolist()
                dw = 1.0 / img_width
                dh = 1.0 / img_height
                x = (coords[0][0] + coords[0][2]) / 2.0
                y = (coords[0][1] + coords[0][3]) / 2.0
                w = coords[0][2] - coords[0][0]
                h = coords[0][3] - coords[0][1]
                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh
                yolo_format = "{} {} {} {} {}\n".format(class_id, x, y, w, h)
                f.write(yolo_format)


    im = Image.fromarray(arr)
    im.save("dataset/train/your_file.jpeg")
    im.save("dataset/train/your_file1.jpeg")

    tr = model.train(epochs=2,batch=1,data='data.yaml',plots=False)
    return tr.loss_items[1]
    print(tr)
    
    


def count_train_folders(directory):
    count = 0
    for folder in os.listdir(directory):
        if "predict" in folder:
            count += 1
    return count

def count_imgs(directory):
    count = 0
    for file in os.listdir(directory):
        if "image" in file:
            count += 1
    return count

run = count_train_folders(os.path.join(dirname, 'runs', 'detect'))

def normalize_batch(x):
    """
    Normalize a batch of tensors along each channel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
    Returns:
        torch.Tensor: Normalized tensor of the same shape as the input.
    """
    mins = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    maxs = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    for i in range(x.shape[0]):
        mins[i] = x[i].min()
        maxs[i] = x[i].max()
    x_ = (x - mins) / (maxs - mins)
    
    return x_

def get_gradient(img, grad_wrt, norm=False, absolute=True, grayscale=False, keepmean=False):
        #print(img,grad_wrt)
        if (grad_wrt.shape != torch.Size([1])) and (grad_wrt.shape != torch.Size([])):
            grad_wrt_outputs = torch.ones_like(grad_wrt).clone().detach()
        else:
           grad_wrt_outputs = None
        print(img,grad_wrt,grad_wrt_outputs)
        attribution_map = torch.autograd.grad(grad_wrt, img, 
                                        grad_outputs=grad_wrt_outputs, 
                                        #allow_unused=True,
                                        # is_grads_batched=True,
                                        # retain_graph=True,
                                        create_graph=True, # Create graph to allow for higher order derivatives but slows down computation significantly
                                        )[0]#.requires_grad_(True)
        print(attribution_map)
        if absolute:
            attribution_map = torch.abs(attribution_map) # attribution_map ** 2 # Take absolute values of gradients
        if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
            attribution_map = torch.sum(attribution_map, 1, keepdim=True)
        if norm:
            if keepmean:
                attmean = torch.mean(attribution_map)
                attmin = torch.min(attribution_map)
                attmax = torch.max(attribution_map)
            attribution_map = normalize_batch(attribution_map) # Normalize attribution maps per image in batch
            if keepmean:
                attribution_map -= attribution_map.mean()
                attribution_map += (attmean / (attmax - attmin))
            
        return attribution_map

def overlay_attr(img, mask, colormap: str = "jet", alpha: float = 0.7):

    cmap = plt.get_cmap(colormap)
    npmask = np.array(mask.clone().detach().cpu().squeeze(0))
    # cmpmask = ((255 * cmap(npmask)[:, :, :3]).astype(np.uint8)).transpose((2, 0, 1))
    cmpmask = (cmap(npmask)[:, :, :3]).transpose((2, 0, 1))
    overlayed_imgnp = ((alpha * (np.asarray(img.clone().detach().cpu())) + (1 - alpha) * cmpmask))
    overlayed_tensor = torch.tensor(overlayed_imgnp, device=img.device)
    
    return overlayed_tensor

def interface_detect(source_im,source_vid,weights,thres,pretrained,user_iou,user_det, get_agnostic,img_size,viz,get_class_name, get_boundingbox):
    print("Source:",source_im)
    
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

    #source = source.as_tensor().requires_grad_(True)
    
    source = source.transpose(2,0,1)
    
    source = torch.tensor(source,requires_grad=True, dtype=torch.float64).unsqueeze(0)
    W = source.shape[2]
    H = source.shape[3]
    W_xtra = source.shape[2] % 32
    H_xtra = source.shape[3] % 32
    source = source[:,:,:W - W_xtra,:H - H_xtra]
    print(source.shape)

    if source is not None:
        if get_class_name == []:
            get_class_name = None
        results = model.predict(source=source,conf=thres/100, iou = user_iou/100,max_det = user_det, agnostic_nms = get_agnostic,imgsz = img_size,visualize = viz,classes = get_class_name,boxes=False,save= True,project = "test")
        loss = my_train(model,DetectionTrainer,source)
        print(loss)
        # print(train)
        # print(results)
        # print("Plot:")
        # print(results[0].plot())
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


        
        


        # Convert the image data to a NumPy array
        # classified = Image.open(os.path.join(dirname ,"Design-User-Interface-YOLOv8", "test","predict" + str(count_train_folders(os.path.join(dirname,"Design-User-Interface-YOLOv8","test"))) , "image0.jpg"))
        # r = np.asarray(classified)
        # r2 = torch.from_numpy(r)
        # res = torch.tensor(r2, requires_grad=True, dtype=torch.float64)
                
        #print(source,)
                
        #res = torch.tensor(results[0].item(),requires_grad=True,dtype=torch.float64)
                
        map = get_gradient(source,loss)
        disp_map = overlay_attr(source, map, colormap="jet", alpha=0.7)



        return [results[0].plot(), None, output_string,disp_map]
            
        

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
