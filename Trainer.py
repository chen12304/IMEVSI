import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model.loss import *
from model.warplayer import warp
import numpy as np
from config import *
import math


def convert(param):
    return {
    k: v
        for k, v in param.items()
        if "module." in k and 'attn_mask' not in k and 'HW' not in k
    }
    

def print_parameter_status(model):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"[可训练] {name}: {param.shape}")
        else:
            print(f"[冻结] {name}: {param.shape}")
    print(f"总参数: {total_params / 1e6:.2f}M")
    print(f"可训练参数占比: {trainable_params / total_params * 100:.2f}%")


class Model:
    def __init__(self, local_rank, resume=None):
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        # 微调时加载权重
        print(self.net)
           
        
                
        self.name = MODEL_CONFIG['LOGNAME']
        self.device()

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.lap = LapLoss_squence()

        self.loss_online = LapLoss(max_levels=1)
        
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        
        # print_parameter_status(self.net)
        # 微调加载模型
        # print('loading pretrained_on_vimeo90k/'+str(299)+'.pkl')
        # self.net.load_state_dict(convert(torch.load('pretrained_on_vimeo90k/'+str(299)+'.pkl', map_location='cuda')))
        
        self.af_down = None
        self.af = None
        self.flow = None
        self.mask = None


        self.af_down_online = None
        self.af_down_online = None
        self.flow_online = None
        self.mask_online = None

        # 继续训练加载权重
        if resume != None:
            print('resuming')
            checkpoint = torch.load(resume, map_location='cuda:{}'.format(local_rank))

            self.net.load_state_dict(convert(checkpoint['model_state_dict']))
            self.optimG.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            self.start_epoch= epoch
            self.optimG.param_groups[0]['capturable'] = True

    
            
    def fineturn_model(self):
        for param in self.net.parameters():
            param.requires_grad = False
        
        for name, param in self.net.block.named_parameters():
            param.requires_grad = True
        for name, param in self.net.feature_bone.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        print_parameter_status(self.net)

    def train(self):
        self.net.train()
    


    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device("cuda"))

    def online_load(self, epoch, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        if rank <= 0 :
            print(f'loading ckpt/{str(epoch)}.pkl')
            self.net.load_state_dict(convert(torch.load(f'ckpt/{str(epoch)}.pkl')))

    def load_model(self, epoch=19, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        def convert_fineturn(param):
            return {
            k: v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if rank <= 0 :
            print(f'loading ckpt/{str(epoch)}.pkl')
            save_dict = convert(torch.load(f'ckpt/{str(epoch)}.pkl'))
            model_dict = self.net.state_dict()
            state_dict = {k:v for k,v in save_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.net.load_state_dict(model_dict)
    
    def save_model(self, rank=0, epoch=0):
        if rank == 0:
            torch.save(self.net.state_dict(),f'ckpt/{str(epoch)}.pkl')

    def save_checkpoint(self, rank=0, epoch=0):
        if rank ==0:
            checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimG.state_dict(),
            'epoch': epoch
            }
            
            torch.save(checkpoint, 'ckpt/checkpoint.pth')
            
    def load_checkpoint(self, filepath):
        def convert(param):
            return {
            k: v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        checkpoint = torch.load(filepath, map_location='cuda')
        self.net.load_state_dict(convert(checkpoint['model_state_dict']))
        self.optimG.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimG.param_groups[0]['capturable'] = True

        epoch = checkpoint['epoch']
        return epoch


    @torch.no_grad()
    def hr_inference_sequence(self, img0=None, img1=None, af_down=None, af=None, down_scale = 1.0, flow = None, mask = None):
        if down_scale!=1:
            if img0!=None:
                img0_down = F.interpolate(img0, scale_factor=down_scale, mode="bilinear", align_corners=False)
            img1_down = F.interpolate(img1, scale_factor=down_scale, mode="bilinear", align_corners=False)
            flow, mask, _ ,self.af_down = self.net.calculate_flow_inference(img0_down, img1_down, af_down)
            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
            self.af, _=self.net.get_afmf(img0, img1, af)
            pred = self.net.coraseWarp_and_Refine(img0, img1, self.af, flow, mask)
            return pred
        else:
            self.flow, self.mask, _ ,self.af = self.net.calculate_flow_inference(img0, img1, af, flow, mask)
            pred = self.net.coraseWarp_and_Refine(img0, img1, self.af, self.flow, self.mask)
            return pred


    def online_hr_inference_sequence(self, img0=None, img1=None, af_down=None, af=None, down_scale = 1.0, flow = None, mask = None):
        if down_scale!=1:
            if img0!=None:
                img0_down = F.interpolate(img0, scale_factor=down_scale, mode="bilinear", align_corners=False)
            img1_down = F.interpolate(img1, scale_factor=down_scale, mode="bilinear", align_corners=False)
            flow, mask, _ ,self.af_down = self.net.calculate_flow_inference(img0_down, img1_down, af_down)
            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
            self.af, _=self.net.get_afmf(img0, img1, af)
            pred = self.net.coraseWarp_and_Refine(img0, img1, self.af, flow, mask)
            return pred
        else:
            self.flow, self.mask, merged ,self.af = self.net.calculate_flow_inference(img0, img1, af, flow, mask)
            pred = self.net.coraseWarp_and_Refine(img0, img1, self.af, self.flow, self.mask)
            return pred
        
    def online_update(self, img0=None, img1=None, af_down=None, af=None, down_scale = 1.0, flow = None, mask = None, learning_rate=1e-5, gt=None, avg_psnr_online=None):
        torch.set_grad_enabled(True)
        assert torch.is_grad_enabled()
        
        
        if down_scale!=1:
            if img0!=None:
                img0_down = F.interpolate(img0, scale_factor=down_scale, mode="bilinear", align_corners=False)
            img1_down = F.interpolate(img1, scale_factor=down_scale, mode="bilinear", align_corners=False)
            flow, mask, _ ,self.af_down_online = self.net.calculate_flow_inference(img0_down, img1_down, af_down)
            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
            self.af_online, _=self.net.get_afmf(img0, img1, af)
            pred = self.net.coraseWarp_and_Refine(img0, img1, self.af_online, flow, mask)
            return pred
        else:
            if flow!=None:
                flow = flow.detach()
                mask = mask.detach()
            self.flow_online, self.mask_online, merged ,self.af_online = self.net.calculate_flow_inference(img0, img1, af, flow, mask)
            pred = self.net.coraseWarp_and_Refine(img0, img1, self.af_online, self.flow_online, self.mask_online)
            for param_group in self.optimG.param_groups:
                param_group['lr'] = learning_rate
            loss_l1 = (self.loss_online(pred, gt)).mean()

            for merge in merged:
                loss_l1 += (self.loss_online(merge, gt)).mean() * 0.5
            # print(loss_l1)
            pred = pred[0].detach().cpu().numpy().transpose(1, 2, 0)
            gt = gt[0].detach().cpu().numpy().transpose(1, 2, 0)
            this_psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
            
            update = True
            if avg_psnr_online==None or this_psnr>avg_psnr_online:
                update=True
            # print(update)
            self.optimG.zero_grad()
            if update:
                loss_l1.backward(retain_graph=True)
                self.optimG.step()

            return this_psnr

        

    
    
    @torch.no_grad()
    def hr_inference(self, img0, img1, TTA = False, down_scale = 1.0, timestep = 0.5, fast_TTA = False):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
            # print(timestep,imgs_down[:,:3].shape,imgs_down[:, 3:6].shape)

            flow, mask, _ = self.net.calculate_flow(imgs_down[:,:3], imgs_down[:, 3:6], timestep)

            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

            af, _ = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs[:,:3], imgs[:,3:6], af, flow, mask)
            return (pred, flow) 

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0][0] + preds[0][1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            pred1 , flow1 = infer(imgs)
            pred2, flow2 = infer(imgs.flip(2).flip(3))

            return (pred1 + pred2.flip(2).flip(3)) / 2
        
    @torch.no_grad()
    def hr_inference_training(self, img0, img1, TTA = False, down_scale = 1.0, timestep = 0.5, fast_TTA = False):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
            # print(timestep,imgs_down[:,:3].shape,imgs_down[:, 3:6].shape)

            flow, mask, _ = self.net.module.calculate_flow(imgs_down[:,:3], imgs_down[:, 3:6], timestep)

            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

            af, _ = self.net.module.feature_bone(img0, img1)
            pred = self.net.module.coraseWarp_and_Refine(imgs[:,:3], imgs[:,3:6], af, flow, mask)
            return (pred, flow) 

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0][0] + preds[0][1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            pred1 , flow1 = infer(imgs)
            pred2, flow2 = infer(imgs.flip(2).flip(3))

            return (pred1 + pred2.flip(2).flip(3)) / 2


    def update(self, imgs, gt, learning_rate=0, training=True, timestep=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            flow, mask, merged, pred = self.net(imgs, timestep=timestep)
            loss_l1 = (self.lap(pred, gt)).mean()

            for merge in merged:
                loss_l1 += (self.lap(merge, gt)).mean() * 0.5

            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1
        else: 
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs)
                return pred, 0
            

    def continous_update(self, imgs, gt=None, learning_rate=0, training=True, timestep=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        
        if training:
            flow, mask, merged, pred = self.net(imgs)
            
            loss_l1 = (self.lap(pred, gt)).mean()
            for i in range(merged.shape[2]):
                loss_l1 += (self.lap(merged[:, :, i], gt)).mean() * 0.5
            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1
        else: 
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs)
                return pred, 0
