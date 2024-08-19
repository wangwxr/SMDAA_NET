# -*- encoding: utf-8 -*-
#Time        :2021/01/04 16:58:26
#Author      :Chen
#FileName    :train_contrast.py
#Version     :1.0

#思路
#准备先加多尺度注意力
import torch
#import _init_paths
import argparse
from plot.evaluate_mu import *
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils.inportant_self_adaption import *
from utils.metric import setup_seed, test
from dataset.lr import adjust_learning_rate
from utils.loss import *
from utils.pixel_contras_loss_new import *
from utils.skeleton import *
from dataset.polyp_dataset import Contrast_dataset, Polyp
from dataset.RIGA_dataloader import *
from models.networks .deeplabv3 import *
from models.SMDAA_net import *
from models.deeplab import *
from utils.visualization import *
from utils.entropy import *
from utils.getlist import *
from models.unet import *
import os
import time
import random
from models.mcdau_backbone.models import MCDAU_Net
from tqdm import tqdm

from dataset.transform import *
import warnings
warnings.filterwarnings('ignore')
from tools.send_email import send_email
from PIL import Image
#按理说batchsize越大，这个对比学习的效果会越好,
BATCH_SIZE = 2
val_batch = 1
NUM_WORKERS = 0
LEARNING_RATE = 0.05
#todo: STARE的学习率
# LEARNING_RATE = 0.01

NUM_STEPS =200
patch_size = (512, 512)
log_name = 'Baseline+DCA+VSG+UPA+PCL'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# dataset = ['hrf']
dataset = ['DRIVE','CHASEDB1']
# dataset = ['DRIVE']
# dataset = ['CHASEDB1']

GPU = '0'
def get_arguments(savepath):
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="sfda_unet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")

    parser.add_argument("--random-mirror", type=bool, default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--snapshot-dir", type=str, default=savepath,
                      help="Where to save snapshots of the model.")

    parser.add_argument("--save-result", type=bool, default=True,
                        help="Whether to save the predictions.")
    # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                  help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--val-batch-size", type=int, default=val_batch)
    return parser.parse_args()


def main(dataset_name):
    TARGET_DATA = fr'D:\2021\wwww\dataset\{dataset_name}'
    AUGMENT_DATA = fr"D:\2021\wwww\dataset\{dataset_name}_diffusion_2"

    SNAPSHOT_DIR = f'../result/{dataset_name}/{log_name}'  # 保存每轮测试图
    write_file = f'{SNAPSHOT_DIR}/metrix_last.txt'
    args = get_arguments(savepath = SNAPSHOT_DIR)
    root_folder = TARGET_DATA


    tr_img_list,tr_label_list,tr_mask_list,ts_img_list,ts_label_list,ts_mask_list = get_list(dataset_name)
    send_string=f'{dataset_name}\n' + f'{log_name} \n'
    seed_torch(20)

    vis = visual(SNAPSHOT_DIR,NUM_STEPS)


    contrast_dataset = RIGA_labeled_set(dataset_name,root_folder, tr_img_list, tr_label_list,tr_mask_list, patch_size, is_diff=True,img_normalize=True)
    ts_dataset = RIGA_labeled_set(dataset_name,root_folder, ts_img_list, ts_label_list, ts_mask_list,patch_size)

    contrast_loader_target  = torch.utils.data.DataLoader(contrast_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=0,
                                                shuffle=True,
                                                pin_memory=True,
                                                collate_fn=source_collate_fn_tr_diff)
                                                          #这个collat_fn是根据contrast_dataset的返回值来的
    test_loader_target = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=args.val_batch_size,
                                                num_workers=0,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    model=SMDAA(in_channels=3, num_classes=1, base_c=32, emb=True).cuda()
    # model=UNet(in_channels=3, num_classes=1, base_c=32).cuda()
    # model=MCDAU_Net( n_classes = 1,emb = True).cuda()
    # model=UNet(contrast=True).cuda()
    # model.load_state_dict(torch.load(args.restore_from))
    # model.load_state_dict(checkpoint['model_state_dict'])

    #TODO:
    optimizer = torch.optim.SGD(
        model.parameters(args), lr=args.learning_rate, momentum=0.99, nesterov=True)
    # optimizer=torch.optim.Adam(model.parameters(args),lr=args.learning_rate,betas=(0.9,0.99))
    #先只算分割的dice损失
    dice_criterion = DiceLoss()
    focal_criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')

    pcl_criterion = PixelContrastLoss(temperature=0.07, # 0.1
                                                base_temperature=0.07,
                                                max_samples=1024,
                                                max_views=1,
                                                device='cuda',
                                                memory=True,
                                                memory_size=100,
                                                pixel_update_freq=10,
                                                pixel_classes=2,
                                                dim=512 )  #todo: 这块改了

    dice_list=[]
    se_list=[]
    sp_list=[]
    acc_list=[]

    for epoch in range(args.num_steps):
        #loss_total是反向传播的那个
        loss_total = torch.tensor(0.0,requires_grad=True)
        seg_loss = []
        loss_compact= []
        loss_aug = []
        loss_target = []
        loss_iou = []
        loss_focal = []
        loss_pcl = []
        loss_sk = []
        tic = time.time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, LEARNING_RATE,args.num_steps)
        for i_iter, batch in enumerate(contrast_loader_target):
            img= torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            no_aug_img = torch.from_numpy(batch['no_aug_img']).cuda().to(dtype=torch.float32)
            no_aug_diff_img = torch.from_numpy(batch['no_aug_diff']).cuda().to(dtype=torch.float32)

            label = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            mask = torch.from_numpy(batch['mask']).cuda().to(dtype=torch.float32)

            normal_img=Variable(no_aug_img).cuda()
            normal_img_diff=Variable(no_aug_diff_img).cuda()

            target_image = Variable(img).cuda()
            label = Variable(label).cuda()
            label_sk = skelton(label) / 255


            global_features ,target_output, emb = model(target_image,feat=True)
            # normal_sfs_3, normal_sfs_2, normal_sfs_1, normal_sfs_0, normal_global_features, normal_decoder_features_0, normal_decoder_features_1,normal_decoder_features_2, normal_decoder_features_3,normal_target_output = model(normal_img,feat=True)



            # normal_diff_sfs_3,normal_diff_sfs_2,normal_diff_sfs_1,normal_diff_sfs_0,normal_diff_global_features,normal_diff_decoder_features_0, normal_diff_decoder_features_1,normal_diff_decoder_features_2, normal_diff_decoder_features_3,normal_diff_target_output=model(normal_img_diff,feat=True)
            # 熵图
            entropy_map = cal_entropy(torch.sigmoid(target_output).cpu().detach().numpy()[:, 0])

            #权重map 之前的weight_map是形如【1，1，1】这种的，现在改成全图
            short_image, long_image, weight_map = cal_wight_map(label.clone(), target_output, entropy_map, BATCH_SIZE,mask)
            temp = 0

            # if epoch==50:
            #     if temp==0:
            #         np.savetxt('output_prob.txt', np.array(target_output[0, 0].cpu().detach()))
            #         np.savetxt("entrop.txt", entropy_map[0])
            #         np.savetxt("weight.txt", weight_map[0])
            #         temp=1

            loss_target_ = dice_criterion(target_output[:, 0], label[:, 0], short_image, long_image,weight_map)
            # loss_target_ = dice_criterion(target_output[:, 0], label[:, 0], short_image, long_image,np.array([1,1,1]))
            loss_iou_ = iou_loss( label[:, 0].cpu().detach().numpy(),target_output[:, 0].cpu().detach().numpy())
            loss_focal_ = focal_criterion(target_output[:, 0],label[:, 0])
            loss_iou.append(loss_iou_)
            loss_focal.append(loss_focal_.cpu().detach().numpy())


            # loss_target_ = dice_criterion(target_output[:, 0], label[:, 0], short_image, long_image, np.array([1,1,1]))

            loss_target.append(loss_target_.item())

            seg_loss_ = loss_target_

            seg_loss.append(seg_loss_.item())

            #3 像素级对比损失
            # 应该把框框去掉 已经去掉了
            contrast_weight = 0.01
            if contrast_weight > 0:  # 两个解码器 一个是cnn分割结果  另一个是transformer分割结果
                # 在我这就是  一个是原图分割结果   另一个是diff  分割结果

                pxlcontrast_loss = torch.tensor(0., requires_grad=True)
                loss_pcl_ = torch.tensor(0., requires_grad=True)
                pxlcontrast_loss = pcl_criterion(
                    emb,
                    label.detach(),
                    (torch.sigmoid(target_output) > 0.5).long(),
                    mask,
                    torch.tensor(entropy_map).unsqueeze(1)
                )
                loss_pcl_ =1 * pxlcontrast_loss
                loss_pcl.append(loss_pcl_.item())
            #4骨架损失
            #加weight 得先sigmoid之后再传进去
            pre_sk = skelton((torch.sigmoid(target_output)>0.5).long()) / 255

            pre_sk = torch.tensor(pre_sk).cuda().float() * (torch.sigmoid(target_output[:,0])).cuda()

            short_ske, long_ske, ske_weight_map = cal_wight_map_ske(torch.tensor(label_sk).cuda(), pre_sk,BATCH_SIZE)
            # #
            loss_sk_img = torch.tensor(0. ,requires_grad=True)
            loss_sk_ = torch.tensor(0. ,requires_grad=True)
            loss_sk_img = dice_criterion(pre_sk.cuda(), torch.tensor(label_sk).cuda(),tiny=short_ske.cpu().numpy(),weight=ske_weight_map.cpu().numpy(),is_sigmoid = False)
            # 0.5 0.5 好像不太行

            # loss_sk_img = dice_criterion(pre_sk.cuda(), torch.tensor(label_sk).cuda(), tiny=short_ske.cpu().numpy(),
            #                              weight=np.array([1,1,1]), is_sigmoid=False)
            loss_sk_ =  1 * loss_sk_img
            loss_sk += [loss_sk_.item()]
            #
            loss_total = seg_loss_ + 0.01 * loss_pcl_ + 0.16 * loss_sk_
            # loss_total = seg_loss_
            # loss_total = seg_loss_ + 0.16* loss_sk_
            #
            # 定义一个变量来保存梯度
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()
            # seg_loss += loss_total.item()


        batch_time = time.time() - tic
        train_info = ('\nEpoch: [{}/{}], Time: {:.2f},'
                      'lr: {:.6f}, Seg Loss: {:.6f},loss_aug：{:.6f},loss_target：{:.6f},loss_compact：{:.6f},losspcl:{:.6f},sk_loss{:.6f}，focal_loss:{:.6f}\n'.format(
            epoch, args.num_steps, batch_time, lr, np.mean(seg_loss), np.mean(loss_aug), np.mean(loss_target),
            np.mean(loss_compact), np.mean(loss_pcl), np.mean(loss_sk),np.mean(loss_focal)))
        print(train_info)
        dice, se, sp, acc, val_info = test(model, test_loader_target, args, dice_criterion, epoch=epoch)
        vis.visual(epoch=epoch, train_dice=1 - np.mean(seg_loss), val_dice=dice.clone().cpu().detach().numpy())
        if epoch == 0:
            with open(write_file, 'a') as f:
                f.write(log_name + '\n')
        with open(write_file, 'a') as f:
            f.write( train_info + val_info + "\n\n")
        dice_list.append(dice.item())
        se_list.append(se.item())
        sp_list.append(sp.item())
        acc_list.append(acc.item())
    print('dice:',dice_list.index(min(dice_list[15:])),min(dice_list[15:]))
    print('se',se_list.index(max(se_list[15:])),max(se_list[15:]))
    print('sp',sp_list.index(max(sp_list[15:])),max(sp_list[15:]))
    print('acc',acc_list.index(max(acc_list[15:])),max(acc_list[15:]))
    with open(write_file,'a') as f:
        f.write(f'best_dice:{min(dice_list[15:])}, {dice_list.index(min(dice_list[15:]))}\n')
        f.write(f'se:{max(se_list[15:])}, {se_list.index(max(se_list[15:]))}\n')
        f.write(f'sp:{max(sp_list[15:])}, {sp_list.index(max(sp_list[15:]))}\n')
        f.write(f'acc:{max(acc_list[15:])}, {acc_list.index(max(acc_list[15:]))}\n')
    send_string+=f'se:{se.item()},sp:{sp.item()},acc:{acc.item()},dice:{dice.item()},lr:{lr}'
    torch.save(model.state_dict(), f'{SNAPSHOT_DIR}/FSM_last.pth')
    return send_string
def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
if __name__ == '__main__':
    for dataset_name in dataset:
        if dataset_name=='STARE':
            LEARNING_RATE = 0.01
        content=main(dataset_name)
        send_email(sendstring=content)


