import sys
from stlpips_pytorch import stlpips
import lpips
import math
from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np
import os

from torchvision.utils import save_image
sys.path.append('.')
from dataset import VimeoDataset_segments
import config as cfg
from Trainer import Model
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--number', default=126,type=int)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
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

data_path = ""
pred_dir = ''
dataset_val = VimeoDataset_segments('test', data_path)
val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=4)

loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")
loss_fn_vgg_st = stlpips.LPIPS(net="vgg", variant="shift_tolerant").to("cuda")
device = torch.device("cuda")

import math
import tqdm

psnr = []
lp = []
stlp = []
ssim = []



os.makedirs(pred_dir, exist_ok=True)

# 外层循环的进度条
pbar = tqdm.tqdm(enumerate(val_data), total=len(val_data), desc="Processing batches")

for batch_idx, imgs in pbar:
    imgs = imgs.to(device, non_blocking=True) / 255.
    imgs, gt = imgs[:, ::2], imgs[:, 1::2]
    
    
    pred, _ = model.continous_update(imgs, gt, training=False)
    for t in range(gt.shape[1]):
        pred_batch = pred[:, t]
        for i in range(gt.shape[0]):
            pred_save_path = os.path.join(pred_dir, f"test_idx{batch_idx*gt.shape[0]+i}_frame{2*t+1}.png")
            save_image(pred_batch[i][None,[2, 1, 0],:,:], pred_save_path)
    
    pred = pred.reshape(-1, pred.shape[2], pred.shape[3], pred.shape[4])
    gt = gt.reshape(-1, gt.shape[2], gt.shape[3], gt.shape[4])
    
    # 计算并记录LP和STLP损失
    lp_value = loss_fn_vgg(gt, pred, normalize=True).mean().cpu().item()
    lp.append(lp_value)
    ssim.append(ssim_matlab(pred, gt).cpu())
    
    stlp_value = loss_fn_vgg_st(gt, pred, normalize=True).mean().cpu().item()
    stlp.append(stlp_value)

    # 计算PSNR并记录
    for j in range(gt.shape[0]):
        psnr_value = -10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item())
        psnr.append(psnr_value)

    # 计算并显示当前的平均PSNR, LP和STLP
    avg_psnr = sum(psnr) / len(psnr)
    avg_lp = sum(lp) / len(lp)
    avg_stlp = sum(stlp) / len(stlp)
    avg_ssim = sum(ssim) / len(ssim)
        
    # 更新进度条并显示平均值
    pbar.set_postfix({'PSNR': avg_psnr, 'LPIPS': avg_lp, 'STLPIPS': avg_stlp, 'SSIM': avg_ssim})



psnr = np.array(psnr).mean()
lp = np.array(lp).mean()
stlp = np.array(stlp).mean()
ssim = np.array(ssim).mean()
print( psnr, lp, stlp, ssim)