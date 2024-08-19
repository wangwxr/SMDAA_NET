# -*- encoding: utf-8 -*-
# Time        :2021/01/04 16:58:26
# Author      :Chen
# FileName    :train_contrast.py
# Version     :1.0

# 思路
import torch
# import _init_paths
import argparse
from plot.evaluate_mu import *
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils.test_predict import setup_seed, test__
from dataset.lr import adjust_learning_rate
from utils.loss import BceDiceLoss, BCELoss, DiceLoss
from dataset.polyp_dataset import Contrast_dataset, Polyp
from dataset.RIGA_dataloader import *
from models.networks.deeplabv3 import *
from models.SMDAA_net import *
from models.deeplab import *
import os
import time
import random
from tqdm import tqdm
import numpy
from dataset.transform import *
import warnings

warnings.filterwarnings('ignore')

from PIL import Image

BATCH_SIZE = 16
NUM_WORKERS = 4
POWER = 0.9
INPUT_SIZE = (512, 512)
# SOURCE_DATA = '/home/cyang/SFDA/data/EndoScene'
# TRAIN_SOURCE_LIST = '/home/cyang/SFDA/dataset/EndoScene_list/train.lst'
# TEST_SOURCE_LIST = '/home/cyang/SFDA/dataset/EndoScene_list/0.01LR.lst'

GPU = '0'
FOLD = 'fold4'
TARGET_MODE = 'gt'
import torchvision
RESTORE_FROM=r'D:\2021\wwww\experiment\abalation\SMDAA_NET\result\CHASEDB1\SMDAA_NET\FSM_last.pth'
SNAPSHOT_DIR = '../predict_test/'
root_folder = r"D:\2021\wwww\dataset\CHASEDB1"

SAVE_RESULT = True
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="sfda_unet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")

    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-result", type=bool, default=True,
                        help="Whether to save the predictions.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                     help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--fold", type=str, default=FOLD,
                        help="choose gpu device.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE,
                        help="choose gpu device.")
    parser.add_argument("--val-batch-size", type=int, default=16)
    return parser.parse_args()


args = get_arguments()


datalist_tsimg = os.listdir(r'D:\2021\wwww\dataset\CHASEDB1\images')
datalist_tslabel = os.listdir(r'D:\2021\wwww\dataset\CHASEDB1\label')
ts_img_list = [os.path.join(r'D:\2021\wwww\dataset\CHASEDB1\images', i) for i in datalist_tsimg]
ts_label_list = [os.path.join(r'D:\2021\wwww\dataset\CHASEDB1\label', i) for i in datalist_tslabel]

patch_size = (512, 512)


def main():
    setup_seed(20)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)

    # 这个collat_fn是根据contrast_dataset的返回值来的
    test_loader_target = torch.utils.data.DataLoader(ts_dataset,
                                                     batch_size=16,
                                                     num_workers=0,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     collate_fn=collate_fn_ts)

    model = SMDAA(in_channels=3, num_classes=1, base_c=32).cuda()
    model.load_state_dict(torch.load(args.restore_from))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # TODO:这块得改
    # optimizer = torch.optim.SGD(
    #     model.parameters(args), lr=args.learning_rate, momentum=0.99, nesterov=True)

    # 先只算分割的dice损失
    dice_criterion = DiceLoss(ignore=255)
    dice, se, sp, acc,val_info = test__(model, test_loader_target, args, dice_criterion,epoch=200)







if __name__ == '__main__':
    main()
