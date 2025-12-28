import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp
from .refine import *



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)
        
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask


class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone
        self.block = nn.ModuleList([Head( kargs['motion_dims'][-1-i] * kargs['depths'][-1-i] + kargs['embed_dims'][-1-i], 
                            kargs['scales'][-1-i], 
                            kargs['hidden_dims'][-1-i],
                            17) 
                            for i in range(self.flow_num_stage)])
        self.unet = Unet(kargs['c'] * 2)

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        return y0, y1

    def calculate_flow(self, img0, img1, timestep, af=None, mf=None):
        # print(timestep)
        mask_list = []
        flow_list = []
        merged = []
        B = img0.size(0)
        flow, mask = None, None
        # appearence_features & motion_features
        if (af is None) or (mf is None):
            af, mf = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).cuda()
            # print((t*mf[-1-i][:B]).shape, ((1-t)*mf[-1-i][B:]).shape, (af[-1-i][:B]).shape, (af[-1-i][B:]).shape)
            # print(torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1).shape)
            if flow != None:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow
                    )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1), 1),
                    None
                    )
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
            

        return flow, mask, merged
    
    def calculate_flow_inference(self, img0=None, img1=None, af=None, flow=None, mask=None):
        mask_list = []
        flow_list = []
        merged = []
        B = img0.size(0)
        if flow == None and mask == None:
            flow = torch.zeros([img0.shape[0], 4, img0.shape[-2], img0.shape[-1]]).cuda()
            mask = (0.5 * torch.ones([img0.shape[0], 1, img0.shape[-2], img0.shape[-1]])).cuda()
        # print(img0.shape, flow.shape)
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])

        # appearence_features & motion_features
        
        if af==None and img0!=None:
            # print(img0.requires_grad, img1.requires_grad)
            af, mf = self.feature_bone(img0, img1)
        else:
            af_back = []
            for appearance in af:
                af_back.append(appearance[B:].detach())
            af, mf = self.feature_bone.recurrent_propagation(img1, af_back)
        for i in range(self.flow_num_stage):
            
            flow_d, mask_d = self.block[i]( torch.cat([mf[-1-i][:B], mf[-1-i][B:], af[-1-i][:B], af[-1-i][B:]],1), 
                                            torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
            if i==0:
                flow = flow_d
                mask = mask_d
            else:
                flow = flow + flow_d
                mask = mask + mask_d
            
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        return flow, mask , merged, af
    
    def get_afmf(self, img0=None, img1=None, af=None):
        B = img0.size(0)
        if af==None and img0!=None:
            af, mf = self.feature_bone(img0, img1)
        else:
            af_back = []
            for appearance in af:
                af_back.append(appearance[B:])
            af, mf = self.feature_bone.recurrent_propagation(img1, af_back)
        return af, mf


    def coraseWarp_and_Refine(self, img0, img1, af, flow, mask):
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = torch.clamp(merged + res, 0, 1)
        return pred


    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def forward(self, x, timestep=0.5):
        pred_sequence = []
        flow_list_sequence = []
        mask_list_sequence = []
        merged_sequnce = []
        af = None
        flow = torch.zeros([x.shape[0], 4, x.shape[-2], x.shape[-1]]).cuda()
        mask = (0.5 * torch.ones([x.shape[0], 1, x.shape[-2], x.shape[-1]])).cuda()
        for frame_num in range(x.size(1)-1):

            img0, img1 = x[:, frame_num], x[:, frame_num+1]
            B = x.size(0)
            flow_list = []
            merged = []
            mask_list = []
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])


            # appearence_features & motion_features
            if af==None:
                af, mf = self.feature_bone(img0, img1)
            else:
                af_back = []
                for appearance in af:
                    af_back.append(appearance[B:])
                af, mf = self.feature_bone.recurrent_propagation(img1, af_back)
            for i in range(self.flow_num_stage):

                
                flow_d, mask_d = self.block[i]( torch.cat([mf[-1-i][:B], mf[-1-i][B:], af[-1-i][:B], af[-1-i][B:]],1), 
                                                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                if i==0:
                    flow = flow_d
                    mask = mask_d
                else:
                    flow = flow + flow_d
                    mask = mask + mask_d
                
                mask_list.append(torch.sigmoid(mask))
                flow_list.append(flow)
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))

            c0, c1 = self.warp_features(af, flow)
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            pred = torch.clamp(merged[-1] + res, 0, 1)
            pred_sequence.append(pred)
            merged = torch.stack(merged, dim=1)
            merged_sequnce.append(merged)
            flow_list_sequence.append(flow_list)
            mask_list_sequence.append(mask_list)
        pred_sequence = torch.stack(pred_sequence, dim=1)
        merged_sequnce = torch.stack(merged_sequnce, dim=1)
        
        return flow_list_sequence, mask_list_sequence, merged_sequnce, pred_sequence
    
