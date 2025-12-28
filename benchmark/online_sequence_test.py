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
from torch.optim import AdamW



from stlpips_pytorch import stlpips
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")
loss_fn_vgg_st = stlpips.LPIPS(net="vgg", variant="shift_tolerant").to("cuda")
device = torch.device("cuda")
lp = []
stlp = []


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--number', default=126,type=int)
parser.add_argument('--save_path', type=str, required=True)
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
model.device()

root_dir = args.path

ssim = []
psnr = []

save_dir = args.save_path

os.makedirs(save_dir, exist_ok=True)

sub_folders = sorted(os.listdir(root_dir))
pbar = tqdm(sub_folders, total=len(sub_folders), desc="Processing subfolders", unit="subfolder")


model.fineturn_model()
model.optimG=AdamW(filter(lambda p: p.requires_grad, model.net.parameters()), lr=1e-5, weight_decay=1e-4)


for subfolder in pbar:

    save_subfloder_path=os.path.join(save_dir, subfolder)
    os.makedirs(save_subfloder_path, exist_ok=True)

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

    # 初始化在线学习的参数
    model.af_down_online = None
    model.af_online = None
    model.flow_online = None
    model.mask_online = None

    last_pred = None
    # 重置优化器，初始化网络参数
    online_psnr=[]
    
    

    for i in tqdm(range(len(gt_files)), desc="Processing frames", leave=False):
        img0_path = os.path.join(subfolder_path, frame_files[i])
        img1_path = os.path.join(subfolder_path, frame_files[i+1])
        gt_path = os.path.join(subfolder_path, gt_files[i])
        # print(img0_path, img1_path, gt_path)

        save_gt_path = os.path.join(save_subfloder_path, gt_files[i])

        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        gt_numpy = cv2.imread(gt_path)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        gt = (torch.tensor(gt_numpy.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
        padder = InputPadder(img1.shape, divisor=32)
        img0, img1 = padder.pad(img0, img1)
        with torch.no_grad():
            pred = model.online_hr_inference_sequence(img0=img0, img1=img1, af_down=model.af_down, af=model.af, down_scale=1, flow=model.flow, mask=model.mask)
        if last_pred==None:
            last_pred = pred
        else:
            avg_psnr_num = None
            if len(online_psnr)!=0:
                avg_psnr_num=sum(online_psnr)/len(online_psnr)

            online_psnr.append(model.online_update(last_pred.detach(), pred.detach(), model.af_down_online, model.af_online, down_scale=1, flow=model.flow_online, mask=model.mask_online, gt=img0, avg_psnr_online=avg_psnr_num))
            last_pred = pred
        pred = padder.unpad(pred[0])
        
        lp_value = loss_fn_vgg(gt, pred.unsqueeze(0), normalize=True).mean().cpu().item()
        lp.append(lp_value)

        stlp_value = loss_fn_vgg_st(gt, pred.unsqueeze(0), normalize=True).mean().cpu().item()
        stlp.append(stlp_value)
               
        ssim.append(ssim_matlab(gt, pred.unsqueeze(0)).detach().cpu().numpy())
        pred = pred.detach().cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(save_gt_path, pred*255)
        gt_numpy = gt_numpy/255.
        psnr.append(-10 * math.log10(((gt_numpy - pred) * (gt_numpy - pred)).mean()))
        avg_ssim = sum(ssim) / len(ssim)
        avg_psnr = sum(psnr) / len(psnr)
        avg_lp = sum(lp) / len(lp)
        avg_stlp = sum(stlp) / len(stlp)

        pbar.set_postfix({'PSNR': avg_psnr, 'SSIM': avg_ssim, 'LPIPS': avg_lp, 'STLPIPS': avg_stlp})
print(sum(psnr) / len(psnr), sum(ssim) / len(ssim), sum(lp) / len(lp), sum(stlp) / len(stlp))


with open("test_metrics.txt", "a") as file:
        file.write(f"{save_dir}, psnr: {sum(psnr) / len(psnr)}, ssim: {sum(ssim) / len(ssim)}, lpips:{sum(lp) / len(lp)}, stlpips:{sum(stlp) / len(stlp)} \n")

