# -*- encoding: utf-8 -*-
#Time        :2020/12/13 19:25:28
#Author      :Chen
#FileName    :load_img.py
#Version     :1.0

import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms

img_size = 512

def load_img_pro(img_path):
    img = Image.open(img_path).convert('RGB')

    img = img.resize((800, 800))
    img = np.array(img).transpose(2,0,1).astype(np.float32)

    # 这种方法改不了img里的值
    # for i in img:
    #     i=(i-i.mean())/i.std()
    for i in range(img.shape[0]):
        img[i]=(img[i]-img[i].mean())/img[i].std()
    return img
#img = img.unsqueeze(0)

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((800, 800))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()