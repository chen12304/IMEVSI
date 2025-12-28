import cv2
import os
import torch, torchvision
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from config import *
# from torchvision.transforms import v2

# from pytorchvideo.data.utils import thwc_to_cthw
import warnings 



warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class VimeoDataset(Dataset):
    def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = model
        self.h = 256
        self.w = 448
        self.data_root = path
        self.image_root = os.path.join(self.data_root, 'GT')#   sequences
        self.mask_root = os.path.join(self.data_root, 'Mask')#   
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')#  tri_trainlist.txt
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')#  tri_testlist.txt
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()                                                    
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        # if self.dataset_name == 'test':
        #    print(self.meta_data[index])
        #    print(index)
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        # print(index)

        '''imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep=0.5
        return img0, gt, img1 ,timestep'''
        # RIFEm with Vimeo-Septuplet
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        ind = [0, 1, 2, 3, 4, 5, 6]
        random.shuffle(ind)
        ind = ind[:3]
        ind.sort()
        img0 = cv2.imread(imgpaths[ind[0]])
        gt = cv2.imread(imgpaths[ind[1]])

        img1 = cv2.imread(imgpaths[ind[2]])        
        timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        return img0, gt, img1 ,timestep
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        # print(mask.shape)
        if 'train' in self.dataset_name:
            img0, gt, img1 = self.aug(img0, gt, img1, 256, 256)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
                timestep = 1-timestep
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]

            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            # print(mask.shape)
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)

                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)

                gt = cv2.rotate(gt, cv2.ROTATE_180)


                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)


                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print(mask.shape)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        # print(mask.shape)

        return [torch.cat((img0, img1, gt), 0), timestep]
    

class VimeoDataset_segments(Dataset):
    def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = model
        self.h = 256
        self.w = 448
        self.data_root = path
        self.image_root = os.path.join(self.data_root, 'GT')#   sequences
        self.mask_root = os.path.join(self.data_root, 'Mask')#   
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')#  tri_trainlist.txt
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')#  tri_testlist.txt
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()                                                    
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs_new = []
        for img in imgs:
            imgs_new.append(img[x:x+h, y:y+w, :])
        return imgs_new
    
    def getimg(self, index):
        # if self.dataset_name == 'test':
        #    print(self.meta_data[index])
        #    print(index)
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        # print(index)

        '''imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep=0.5
        return img0, gt, img1 ,timestep'''
        # RIFEm with Vimeo-Septuplet
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        ind = [0, 1, 2, 3, 4, 5, 6]
        img0 = cv2.imread(imgpaths[ind[0]])
        img1 = cv2.imread(imgpaths[ind[1]])

        img2 = cv2.imread(imgpaths[ind[2]])        
        img3 = cv2.imread(imgpaths[ind[3]])
        img4 = cv2.imread(imgpaths[ind[4]])
        img5 = cv2.imread(imgpaths[ind[5]])
        img6 = cv2.imread(imgpaths[ind[6]])
        return img0, img1,img2,img3,img4,img5,img6
            
    def __getitem__(self, index):        
        img0, img1,img2,img3,img4,img5,img6 = self.getimg(index)
        # print(mask.shape)
        # img0, img1,img2,img3,img4,img5,img6 = self.aug([img0, img1,img2,img3,img4,img5,img6], 256, 256)
        if 'train' in self.dataset_name:
            img0, img1,img2,img3,img4,img5,img6 = self.aug([img0, img1,img2,img3,img4,img5,img6], 256, 256)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                img2 = img2[:, :, ::-1]
                img3 = img3[:, :, ::-1]
                img4 = img4[:, :, ::-1]
                img5 = img5[:, :, ::-1]
                img6 = img6[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                img3 = img3[:, ::-1]
                img4 = img4[:, ::-1]
                img5 = img5[:, ::-1]
                img6 = img6[:, ::-1]
            # print(mask.shape)
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
                img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
                img3 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
                img4 = cv2.rotate(img4, cv2.ROTATE_90_CLOCKWISE)
                img5 = cv2.rotate(img5, cv2.ROTATE_90_CLOCKWISE)
                img6 = cv2.rotate(img6, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
                img2 = cv2.rotate(img2, cv2.ROTATE_180)
                img3 = cv2.rotate(img3, cv2.ROTATE_180)
                img4 = cv2.rotate(img4, cv2.ROTATE_180)
                img5 = cv2.rotate(img5, cv2.ROTATE_180)
                img6 = cv2.rotate(img6, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img3 = cv2.rotate(img3, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img4 = cv2.rotate(img4, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img5 = cv2.rotate(img5, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img6 = cv2.rotate(img6, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print(mask.shape)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1)
        img3 = torch.from_numpy(img3.copy()).permute(2, 0, 1)
        img4 = torch.from_numpy(img4.copy()).permute(2, 0, 1)
        img5 = torch.from_numpy(img5.copy()).permute(2, 0, 1)
        img6 = torch.from_numpy(img6.copy()).permute(2, 0, 1)
        # print(mask.shape)
        return torch.stack((img0, img1, img2, img3,img4, img5,img6), 0)
    


class Vimeo_Tri_Dataset(Dataset):
    def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = model
        self.h = 256
        self.w = 448
        self.data_root = path
        self.image_root = os.path.join(self.data_root, 'sequences')#   sequences
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')#  tri_trainlist.txt
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')#  tri_testlist.txt
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()                                                    
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        # if self.dataset_name == 'test':
        #    print(self.meta_data[index])
        #    print(index)
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        # print(index)

        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep=0.5
        return img0, gt, img1 ,timestep
        
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        # print(mask.shape)
        if 'train' in self.dataset_name:
            img0, gt, img1 = self.aug(img0, gt, img1, 256, 256)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
                timestep = 1-timestep
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]

            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            # print(mask.shape)
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)

                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)

                gt = cv2.rotate(gt, cv2.ROTATE_180)


                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)


                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print(mask.shape)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        # print(mask.shape)

        return [torch.cat((img0, img1, gt), 0), timestep]


class SnuFilm(Dataset):
    def __init__(self, data_root, data_type="extreme"):
        self.data_root = data_root
        self.data_type = data_type
        
        self.load_data()


    def __len__(self):
        return len(self.meta_data)


    def load_data(self):
        if self.data_type == "easy":
            easy_file = os.path.join(self.data_root, "eval_modes/test-easy.txt")
            with open(easy_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "medium":
            medium_file = os.path.join(self.data_root, "eval_modes/test-medium.txt")
            with open(medium_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "hard":
            hard_file = os.path.join(self.data_root, "eval_modes/test-hard.txt")
            with open(hard_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "extreme":
            extreme_file = os.path.join(self.data_root, "eval_modes/test-extreme.txt")
            with open(extreme_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "extreme_l":
            extreme_file = os.path.join(self.data_root, "eval_modes/top-half-motion-sufficiency_test-extreme.txt")
            with open(extreme_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "hard_l":
            extreme_file = os.path.join(self.data_root, "eval_modes/top-half-motion-sufficiency_test-hard.txt")
            with open(extreme_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        


    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = imgpath.split()

        # Load images
        img0 = cv2.imread(os.path.join(self.data_root, imgpaths[0]))
        gt = cv2.imread(os.path.join(self.data_root, imgpaths[1]))
        img1 = cv2.imread(os.path.join(self.data_root, imgpaths[2]))

        return img0, gt, img1


    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)



if __name__=='__main__':
    import time
    path = '/media/gpu/CKY/vfidataset/LAVIB'
    pipe = LAVIBPipline('lavib', path=path, batch_size=8, num_threads=2, device_id=0)
    pipe.get_reader()
    pipe.build()

    time_start = time.time()
    for _ in range(10):
        time_1 = time.time()
        pipe_out = pipe.run()
        sequence_out = pipe_out[0].as_cpu().as_array()
        print(f"GPU 0 output shape: {sequence_out.shape}")
        print(time.time()-time_1)
    print(time.time()-time_start)
    pipe = LAVIBPipline('lavib', path=path, batch_size=8, num_threads=2, device_id=0)
    pipe.get_reader(length=15)
    pipe.build()
    
    time_start = time.time()
    for _ in range(10):
        time_1 = time.time()
        pipe_out = pipe.run()
        sequence_out = pipe_out[0].as_cpu().as_array()
        print(f"GPU 0 output shape: {sequence_out.shape}")
        print(time.time()-time_1)
    print(time.time()-time_start)

    