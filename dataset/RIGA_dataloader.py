import os

from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *


class RIGA_labeled_set(data.Dataset):
    def __init__(self, db,root, img_list, label_list, mask_list=[],target_size=(512, 512), img_normalize=True, is_diff=False,mean=0,std=0,dif_mean=0,dif_std=0):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.mask_list = mask_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize
        self.is_diff = is_diff
        self.db = db
    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        label_file = join(self.root, self.label_list[item])
        mask_file = join(self.root, self.mask_list[item])
        img = Image.open(img_file)
        label = Image.open(label_file)
        mask = Image.open(mask_file).convert('L')
        img = img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        mask = mask.resize(self.target_size, resample=Image.NEAREST)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        mask_npy = np.array(mask)
        mask_mask = np.zeros_like(mask_npy)
        mask_mask[mask_npy > 0] = 1

        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                mean = img_npy[i][mask_mask==1].mean()
                std = img_npy[i][mask_mask==1].std()
                img_npy[i] = (img_npy[i] - mean) / std
                img_npy[i] = img_npy[i] * mask_mask

        label_npy = np.array(label)
        label_mask = np.zeros_like(label_npy)
        label_mask[label_npy > 0] = 1


        if self.is_diff:
            if self.db =='DRIVE':
                diff_img_file = join(self.root, self.img_list[item]).replace(f'{self.db}',f'{self.db}_diffusion_2')
            elif self.db =='CHASEDB1':
                diff_img_file = join(self.root, self.img_list[item]).replace(f'{self.db}',f'{self.db}_diffusion_2').replace('jpg','tif')
            elif self.db == 'hrf':
                diff_img_file = join(self.root, self.img_list[item]).replace(f'{self.db}',f'{self.db}_diffusion_2').replace('jpg','tif').replace('JPG','tif')
            elif self.db == 'STARE':
                diff_img_file = join(self.root, self.img_list[item]).replace(f'{self.db}',f'{self.db}_diffusion_2').replace('png','tif')

            diff_img = Image.open(diff_img_file)
            diff_img = diff_img.resize(self.target_size)
            diff_img_npy = np.array(diff_img).transpose(2, 0, 1).astype(np.float32)
            if self.img_normalize:
                for i in range(diff_img_npy.shape[0]):
                    mean = diff_img_npy[i][mask_mask == 1].mean()
                    std = diff_img_npy[i][mask_mask == 1].std()
                    diff_img_npy[i] = (diff_img_npy[i] - mean) / std
                    diff_img_npy[i] = diff_img_npy[i] * mask_mask
            return img_npy, label_mask[np.newaxis], img_file,diff_img_npy,mask_mask

        return img_npy, label_mask[np.newaxis], img_file,mask_mask

    def __len__(self):
        return self.len





