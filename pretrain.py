
import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import torch.distributed
from tqdm import tqdm
# from stlpips_pytorch import stlpips
# import lpips

import torchvision.transforms as transforms

from Trainer import Model

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from config import *

from dataset import VimeoDataset
from dataset import VimeoDataset_segments


exp = os.path.abspath('.').split('/')[-1]

def calc_psnr(pred, gt, mask=None):
    '''
        Here we assume quantized(0-1.) arguments.
    '''
    diff = (pred - gt)

    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean() + 1e-8    # mse can (surprisingly!) reach 0, which results in math domain error

    return -10 * math.log10(mse)

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        return 1e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (1e-4 - 1e-5) * mul + 1e-5

def train(model, local_rank, batch_size, data_path, tset, GPU_num=1, resume=None):
    if local_rank == 0:
        writer = SummaryWriter('log/train_pretained')
    device = torch.device("cuda:{}".format(str(local_rank)))
    step = 0
    nr_eval = 0
    best = 0
    dataset = VimeoDataset_segments('train', data_path)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    # dataset_val = SnuFilm(data_root="/media/gpu/CKY/vfidataset/SNUFILM/", data_type="hard")
    dataset_val = VimeoDataset_segments('test', data_path)
    sampler_val = DistributedSampler(dataset_val)
    val_data = DataLoader(dataset_val, batch_size=4, pin_memory=True, num_workers=4,sampler=sampler_val)
    print('training...')
    time_stamp = time.time()
    best_psnr=0
    best_epoch=0
    loss_list=[]
    start_epoch=0
    resume = 'None'
    if resume!='None':
        print('resuming')
        start_epoch = model.load_checkpoint(resume)
        start_epoch+=1
        step = args.step_per_epoch*start_epoch
        nr_eval=start_epoch
    for epoch in range(start_epoch,300):
        # length = random.randint(5,10)
        # if local_rank == 0:
        #     psnr=evaluate(model, val_data, nr_eval, local_rank, writer=writer)
        sampler.set_epoch(epoch)
        bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]{postfix}'
        iter_length = args.step_per_epoch
    
        with tqdm(total=iter_length,bar_format=bformat,ascii='░▒█',miniters=1) as pbar:
            for i, imgs in enumerate(train_data):
                # psnr=evaluate(model, val_data, nr_eval, local_rank)
                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                # mask = imgs[2].to(device, non_blocking=True)/ 255.
                
                # print(imgs)
                
                # print(mask.shape)
                imgs = imgs.to(device, non_blocking=True) / 255.
                imgs, gt = imgs[:, ::2], imgs[:, 1::2]
                learning_rate = get_learning_rate(step)
                _, loss = model.continous_update(imgs, gt, learning_rate, training=True)
                # _, loss = model.mult_frame_update(imgs, learning_rate=learning_rate, training = True)
                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                if local_rank == 0:
                    pbar.set_postfix_str('epoch:{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch, data_time_interval, train_time_interval, loss))
                    pbar.update()
                    writer.add_scalar('Train/Loss', loss, step)
                    writer.add_scalar('Train/LearningRate', learning_rate, step)
                step += 1
        # train_data.reset()
        #torch.cuda.empty_cache()
        if nr_eval % 1 == 0 and local_rank==0:
            
            
            evaluate(model, val_data, nr_eval, local_rank, writer)
        if local_rank==0:    
            model.save_model(local_rank, epoch)
            model.save_checkpoint(local_rank, nr_eval)
        nr_eval += 1
            
            
        #dist.barrier()
@torch.no_grad()
def evaluate(model, val_data, nr_eval, local_rank, writer=None):
    # loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")
    # loss_fn_vgg_st = stlpips.LPIPS(net="vgg", variant="shift_tolerant").to("cuda")
    device = torch.device("cuda:{}".format(str(local_rank)))
    psnr = []
    lp = []
    stlp = []
    for _, imgs in enumerate(val_data):
        # mask = imgs[2].to(device, non_blocking=True)/ 255.
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, ::2], imgs[:, 1::2]
        pred, loss = model.continous_update(imgs, gt, training=False)
        pred = pred.reshape(-1, pred.shape[2], pred.shape[3], pred.shape[4])
        gt = gt.reshape(-1, gt.shape[2], gt.shape[3], gt.shape[4])
        # lp.append(loss_fn_vgg(gt, pred, normalize=True).mean().cpu().item())
        # stlp.append(loss_fn_vgg_st(gt, pred, normalize=True).mean().cpu().item())
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
    psnr = np.array(psnr).mean()
    # lp = np.array(lp).mean()
    # stlp = np.array(stlp).mean()
    if local_rank == 0:
        print(str(nr_eval), psnr, lp, stlp)
        writer.add_scalar('psnr', psnr, nr_eval)
        # writer.add_scalar('lpips', lp, nr_eval)
        # writer.add_scalar('stlpips', stlp, nr_eval)
        return psnr            
            
            
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--data_path', type=str, help='data path of lavib', default='/media/SCRATCH/LAVIB')
    parser.add_argument("--pretrained" , type=str, help="Load from a pretrained model.")
    parser.add_argument('--resume', type=str, help='Path to the checkpoint to resume training from', default='None')
    
    args = parser.parse_args()
    print(os.getenv('LOCAL_RANK', -1))
    args.local_rank=int(eval(os.getenv('LOCAL_RANK', -1)))
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size, rank=args.local_rank)
    
    print(args.local_rank,type(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    GPU_num = args.world_size
    
    if args.pretrained:
        ## For low data, it is better to load from a supervised pretrained model
        loadStateDict = torch.load(args.pretrained)
        modelStateDict = model.net.state_dict()
        for k,v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                print("Loading " , k)
                modelStateDict[k] = v
            else:
                print("Not loading" , k)
        model.net.load_state_dict(modelStateDict)
    
    train(model, args.local_rank, args.batch_size, args.data_path, args.set, GPU_num, args.resume)
