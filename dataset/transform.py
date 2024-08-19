# -*- coding:utf-8 -*-
import os
from copy import deepcopy
import numpy as np
from PIL import Image
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from tools.fourier import FDA_source_to_target_np

def fourier_augmentation(data, fda_beta=0.15):
    this_fda_beta = round(np.random.random() * fda_beta, 2)
    lowf_batch = np.random.permutation(data)
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data

def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def source_collate_fn_tr(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict
def source_collate_fn_tr_diff(batch):
    image, label, name, diff_image, mask = zip(*batch)
    no_aug_image=deepcopy(image)
    no_aug_image=np.stack(no_aug_image,0)
    no_aug_diff=deepcopy(diff_image)
    no_aug_diff=np.stack(no_aug_diff,0)

    image = np.stack(image, 0)
    label = np.stack(label, 0)
    mask = np.stack(mask, 0)
    name = np.stack(name, 0)
    diff_image=np.stack(diff_image,0)
    cat_img=np.concatenate((image,diff_image),axis=1)
    cat_label=np.concatenate((label,label),axis=0)
    cat_name=np.concatenate((name,name),axis=0)

    data_dict = {'data': image, 'seg': label, 'name': name,'no_aug_img':no_aug_image,'no_aug_diff':no_aug_diff}
    # tr_transforms = get_train_transform()
    # data_dict = tr_transforms(**data_dict)

    # diff_data_dict = {'data': diff_image, 'seg': label, 'name': name}
    diff_data_dict = {'data': cat_img, 'seg':  np.concatenate((label,np.expand_dims(mask, axis=1)),axis=1), 'name': name}
    diff_tr_transforms = get_train_transform()
    diff_data_dict = diff_tr_transforms(**diff_data_dict)
    # os.makedirs(r'D:\2021\wwww\experiment\base_UNET\dual_baseline_diff1',exist_ok=True)
    # for i in range(name.shape[0]):
    #     Image.fromarray(np.uint8(diff_data_dict['data'][i][0:3].transpose(1,2,0))).save(r'D:\2021\wwww\experiment\base_UNET\dual_baseline_diff1\\'+name[i].split('\\')[-1])
    #     Image.fromarray(np.uint8(diff_data_dict['data'][i][3:].transpose(1,2,0))).save(r'D:\2021\wwww\experiment\base_UNET\dual_baseline_diff1\\'+name[i].split('\\')[-1].replace('.tif','-1.tif'))

    # data_dict['diff_data'] = diff_data_dict['data']

    data_dict['diff_data'] = diff_data_dict['data'][:,3:]
    data_dict['data'] = diff_data_dict['data'][:,0:3]
    data_dict['seg'] = np.expand_dims(diff_data_dict['seg'][:,0],axis=1)
    data_dict['mask'] = np.expand_dims(diff_data_dict['seg'][:,1],axis=1)
    return data_dict


def source_collate_fn_tr_fda(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    fda_data = fourier_augmentation(data_dict['data'])
    data_dict['fda_data'] = fda_data
    return data_dict


def collate_fn_ts(batch):
    image, label, name, mask= zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    return data_dict


def collate_fn_tr(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'name': name}
    return data_dict
