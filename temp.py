import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
import os, os.path
import random
import torch
import matplotlib.pyplot as plt
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import (RANK, DEFAULT_CFG)
from PIL import Image

class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',14: 'bird', 15: 'cat',16: 'dog',17: 'horse',18: 'sheep',19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',69: 'oven',70: 'toaster',71: 'sink',72: 'refrigerator',73: 'book',74: 'clock',75: 'vase',76: 'scissors',77: 'teddy bear',78: 'hair drier',79: 'toothbrush'}

def my_train(model,trainer,imgt):
    with torch.autograd.enable_grad():
        #model = YOLO(model)
        #model.trainer = trainer
        results = model.predict(imgt)
        for param in model.parameters():
            param.requires_grad = True
        for result in results:
            open('dataset\\train\labels\your_file.txt', 'w').close()
            with open('dataset\\train\labels\your_file.txt', 'a') as f:
                for box in result.boxes:
                    class_id = int(box.cls)
                    coords = box.xywhn.cpu().tolist()
                    x = coords[0][0]
                    y = coords[0][1]
                    w = coords[0][2]
                    h = coords[0][3]
                    yolo_format = "{} {} {} {} {}\n".format(class_id, x, y, w, h)
                    f.write(yolo_format)

        for result in results:
            open('dataset\\train\labels\your_file1.txt', 'w').close()
            with open('dataset\\train\labels\your_file1.txt', 'a') as f:
                for box in result.boxes:
                    class_id = int(box.cls)
                    coords = box.xywhn.cpu().tolist()
                    x = coords[0][0]
                    y = coords[0][1]
                    w = coords[0][2]
                    h = coords[0][3]
                    yolo_format = "{} {} {} {} {}\n".format(class_id, x, y, w, h)
                    f.write(yolo_format)

    
        
        arr = results[0].plot()

        im = Image.fromarray(arr)
        im.save("dataset/train/images/your_file.jpeg")
        im.save("dataset/train/images/your_file1.jpeg")

        tr = model.train(epochs=1,data='data.yaml')
        x = model.trainer.loss
        return x
        #print(tr)
    
    



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






#interface_detect(arr,None,"yolov8n.pt",50,"yolov8n.pt",50,50, False,640,False,None,True)

def main():

    image = Image.open("park.jpeg")

    arr = np.array(image)
        
    arr = arr.transpose(2,0,1)
    source= torch.tensor(arr,requires_grad=True, dtype=torch.float64).unsqueeze(0).cuda()
    W = source.shape[2]
    H = source.shape[3]
    W_xtra = source.shape[2] % 32
    H_xtra = source.shape[3] % 32
    source = source[:,:,:W - W_xtra,:H - H_xtra]

    class_loss = my_train(YOLO('yolov8n.pt'),DetectionTrainer,source)
#     class_loss = source.clone()
#     class_loss = temp.clone()
# # Copy the gradients from temp to class_loss
#     class_loss.grad = source.grad

    grad = get_gradient(source,class_loss)
    overlayed = overlay_attr(source, grad)
    cv2.imshow("overlayed", overlayed)

if __name__ == "__main__":
    main()