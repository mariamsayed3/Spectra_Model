#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:14:29 2020
@author: Boxiang Yun   School:ECNU&HFUT   Email:971950297@qq.com
"""
from torch.utils.data.dataset import Dataset
import skimage.io
#from skimage.metrics import normalized_mutual_information
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import cv2
import os
from argument import Transform
from spectral import *
from spectral import open_image
import random
import math
from scipy.ndimage import zoom
import warnings
import tiff
warnings.filterwarnings('ignore')
from einops import repeat

class Data_Generate_Cho(Dataset):#
    def __init__(self, img_paths, seg_paths=None,
                 cutting=None, transform=None,
                 channels=None, outtype='3d', envi_type='img',
                 multi_class= 1):
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.transform = transform
        self.cutting = cutting
        self.channels = channels
        self.outtype = outtype
        self.envi_type = envi_type
        self.multi_class = multi_class

    def __getitem__(self,index):
        img_path = self.img_paths[index]
        mask_path = self.seg_paths[index]
        if mask_path.endswith('.npz'):
            mask = np.load(mask_path)['gt']
            mask[mask > 8] = 0
        elif mask_path.endswith('.hdr'):
            mask = envi.open(mask_path, image=mask_path.replace('.hdr', ''))[:, :, 0]
            mask = np.squeeze(mask)
        elif mask_path.endswith('.tif'):
            masks = tiff.read_mtiff(mask_path)
            y_seg = tiff.mtiff_to_2d_arr(masks)
            mask = cv2.resize(y_seg, (320, 256), interpolation=cv2.INTER_NEAREST)
        elif mask_path.endswith('.npy'):
            print("hello from mask")
            mask = np.load(mask_path) 
            print(mask.shape[0])
            print(mask.shape[1])       
        else:
            mask = (cv2.imread(mask_path, 0) / 255).astype(np.uint8)

        if img_path.endswith('.hdr'):
            img = envi.open(img_path, image=img_path.replace('.hdr', self.envi_type))[:, :, :]
        elif img_path.endswith('.npy'):
            print("hello_2 in image")
            img = np.load(img_path)
            
        elif img_path.endswith('.tif'):
            x, _, _, _ = tiff.read_stiff(img_path)
            x = cv2.resize(x, (320, 256), interpolation=cv2.INTER_NEAREST)
            chosen_channels = np.linspace(0, x.shape[2] - 1, num=51, dtype=int)
            new_x = [x[:, :, channel] for channel in chosen_channels]
            img = np.stack(new_x, axis=2)

        if mask_path.endswith('.hdr'):
            h = 320
            selected_channels = np.linspace(0, img.shape[2] - 1, 300, dtype=int)
            img = np.stack([img[:, :, i] for i in selected_channels], axis=-1)
            img = cv2.resize(img, (h, h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (h, h), interpolation=cv2.INTER_NEAREST)
            

        img = img[:, :, self.channels] if self.channels is not None else img

        # if mask_path.endswith('.npz'):
        #     mask = np.delete(mask, 445, 0)
        #     img = np.delete(img, 445, 0)

        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")

        if img.shape != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        if self.transform != None:
            if img.dtype != np.uint8:
                print("hii from main if image")
                if img.max() <= 1.0:  # Assuming float images in range [0, 1]
                    print("hii from if image")
                    img = (img * 255).astype(np.uint8)
                else:
                    print("hiii from else image")
                    img = img.astype(np.uint8)
        
            if mask.dtype not in [np.uint8, np.int32]:
                print("hii from main if mask")
                if mask.max() <= 255:
                    print("hiii from mask if")
                    mask = mask.astype(np.uint8)
                else:
                    print("hiiii from mask else")
                    mask = mask.astype(np.int32)
            print("before transform")
            img, mask = self.transform((img, mask))
            print("afterr transform")
        mask = mask.astype(np.uint8)
        if self.cutting is not None:
            while(1):
                xx = random.randint(0, img.shape[0] - self.cutting)
                yy = random.randint(0, img.shape[1] - self.cutting)
                patch_img = img[xx:xx + self.cutting, yy:yy + self.cutting]
                patch_mask = mask[xx:xx + self.cutting, yy:yy + self.cutting]
                if patch_mask.sum()!=0: break
            img = patch_img
            mask = patch_mask


        img = img[:, :, None] if len(img.shape)==2 else img
        img = np.transpose(img, (2, 0, 1))
        if self.outtype == '3d':
            img = img[None]
        mask = mask[None, ].astype(np.float32)
        img = img.astype(np.float32)
        print("end")
        return img, mask
            
    def __len__(self):
        return len(self.img_paths)