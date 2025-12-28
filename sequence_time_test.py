import os
import sys
import cv2
import math
import skimage
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab


lp = []
stlp = []

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--number', default=122,type=int)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
down_scale = 1
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )

model = Model(-1)
model.load_model(epoch=args.number)
model.eval()
model.device()

root_dir = args.path

ssim = []
psnr = []




sub_folders = sorted(os.listdir(root_dir))
pbar = tqdm(sub_folders, total=len(sub_folders), desc="Processing subfolders", unit="subfolder")

import time

start_time = None

for subfolder in pbar:


    subfolder_path = os.path.join(root_dir, subfolder)
    if os.path.isdir(subfolder_path):
        img_files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    
    frame_files = img_files[::2]
    gt_files = img_files[1::2]
    if len(gt_files)==len(frame_files):
        gt_files = gt_files[:-1]
    img0 = None
    img1 = None
    model.af_down = None
    model.af = None
    model.flow = None
    model.mask = None
    
    start_time = time.time()
    
    for i in tqdm(range(len(gt_files)), desc="Processing frames", leave=False):
        img0_path = os.path.join(subfolder_path, frame_files[i])
        img1_path = os.path.join(subfolder_path, frame_files[i+1])


        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        

        img0 = (torch.tensor(img0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        

        padder = InputPadder(img1.shape, divisor=32)
        img0, img1 = padder.pad(img0, img1)
        
        
        pred = model.hr_inference_sequence(img0=img0, img1=img1, af_down=model.af_down, af=model.af, down_scale=1, flow=model.flow, mask=model.mask)[0]
        pred = padder.unpad(pred)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time/549:.6f} 秒")
        