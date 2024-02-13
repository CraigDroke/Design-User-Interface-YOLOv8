import logging
import gradio as gr
from pathlib import Path
import sys
import os
import torch
import wandb
import subprocess

sys.path.append('Interface_Dependencies')
sys.path.append('Interface')
sys.path.append('yoloV7train')
sys.path.append('models')
sys.path.append('./') 

#from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
from models.experimental import attempt_load
from models.yolo import Model
from parser_1 import create_parser
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from train import train
import yaml
from utils.google_utils import attempt_download 
import argparse

shared_theme = gr.themes.Base()

logger = logging.getLogger(__name__)



def interface_login(logger):
    if logger == 'WANDB':
        result = wandb.login()
        if result:
            gr.Info("Logged in to WANDB")
        else:
            gr.Warning("Failed to log in to WANDB")
    elif logger == 'ClearML':
        pass
    elif logger == 'Tensorboard':
        pass

def interface_finetune(hyp, opt, weights, device, rank, nc):
    # Load a pretrained YOLOv7 model
    opt = create_parser()

    with open(opt.hyp, custom_hypfile) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    hyp = yaml.load('coco.yaml')
    custom_hypfile = hyp


    device = select_device(opt.device, default_device, batch_size=opt.batch_size)
    default_device = select_device('cpu')
    device = select_device()
    
    pretrained = weights.endswith('.pt')
    if pretrained:
            with torch_distributed_zero_first(rank):
                attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))
    model = opt.weights('yolov7.pt')  # Load an official Detect model
    return model

def interface_train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze   
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check 

    opt = create_parser()
    # Hyperparameters
    

    

# def interface_train(hyp ,opt , device, data):
#     # model = ('coco128.yaml')
#     # if is_fintune:
#     #     model = interface_finetune()

#     opt = create_parser()
#     # Hyperparameters
#     with open(opt.hyp) as f:
#         hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
#     device = select_device(opt.device, batch_size=opt.batch_size)
#     train.train(hyp, opt, device)#data=dataset, epochs=epochs, imgsz=imgsz)
    
def interface_train_wandb(cfg, weights, data, rank, device, opt, hyp, epochs, batch_size, img_size, resume, evolve, project_name, model_name):
    # Step 1: Initialize a Weights & Biases run
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming
    wandb.init(project=project_name, job_type="training")
    model = weights(f"{model_name}.pt")


    # Step 3: Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Step 4: Train
    model.train(cfg, weights, data, hyp, epochs, batch_size, img_size, resume, evolve)

    # Step 5: Validate the Model
    model.val()

    # # Step 6: Perform Inference and Log Results
    # model(["Images\Craig.jpg", "Images\WalterWhite.jpg"])

    # Step 7: Finalize the W&B Run
    wandb.finish()

# Define a function to start training with the provided hyperparameters
def start_training(cfg, weights, data, hyp, epochs, batch_size, img_size, resume, evolve):
    # Construct the training command based on input parameters
    training_command = f"python train.py --cfg {cfg} --weights {weights} --data {data} --hyp {hyp} --epochs {epochs} --batch-size {batch_size} --img-size {img_size} --resume {resume} --evolve {evolve}"

    # Start training using subprocess
    subprocess.run(training_command, shell=True)
