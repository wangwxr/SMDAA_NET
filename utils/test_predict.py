# -*- encoding: utf-8 -*-
#Time        :2020/12/19 21:17:58
#Author      :Chen
#FileName    :metric.py
#Version     :1.0

from utils.skeleton import *
import torch
import numpy as np
import math
import random
import os
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import matplotlib.pyplot as plt
unloader = transforms.ToPILImage()
from PIL import Image,ImageDraw
from utils.eval import ConfusionMatrix,DiceCoefficient,MetricLogger
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def tensor_to_PIL(tensor, is_trans = False):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image



def test__(model, dataloader, args, dice_criterion,epoch=0):

    confmat = ConfusionMatrix(2)
    # dice = DiceCoefficient(num_classes=2, ignore_index=255)
    dice_loss=0
    Auc=0
    predict_epoch=os.path.join('../predict_root',str(epoch))

    os.makedirs(predict_epoch,exist_ok=True)
    header = 'Test:'
    model.eval()
    with torch.no_grad():
        total_batch = len(dataloader.dataset)
        print('begin 0.01LR')
        bar = tqdm(enumerate(dataloader), total=math.ceil(total_batch/args.val_batch_size))
        for i, data in bar:
            img = torch.from_numpy(data['data']).cuda().to(dtype=torch.float32)
            gt = torch.from_numpy(data['seg']).cuda().to(dtype=torch.float32)

            # img, gt ,mask= data['image'], data['label'],data['mask']

            img = Variable(img).cuda()

            #_, output = model(img)
            output_ = model(img)
            output=torch.sigmoid(output_)

            for j in range(output.shape[0]):
                seg_result=np.zeros((512,512))
                seg_result[output.cpu()[j].squeeze(0)>0.5]=255
                photo=Image.fromarray(seg_result.astype(np.uint8))
                photo.save(os.path.join(predict_epoch,data['name'][j].split('\\')[-1]))
                #若是保存骨架化的 直接解注释就行
                # result_sk = skelton(torch.tensor(seg_result).unsqueeze(0).unsqueeze(0))
                # result_sk_photo = Image.fromarray(result_sk.squeeze(0).astype(np.uint8))
                # result_sk_photo.save(os.path.join(predict_epoch,'ske'+data['name'][j].split('\\')[-1]))

                diff = (seg_result - gt[j].cpu().numpy()*255).squeeze(0)
                #>0多分了 <0少分了
                # 创建一个PIL Image对象作为标记
                output_colored=np.concatenate((seg_result[np.newaxis,:],seg_result[np.newaxis,:],seg_result[np.newaxis,:]),axis=0)

                for w in range(seg_result.shape[0]):
                    for h in range(seg_result.shape[1]):
                        if diff[w][h] > 0:
                            output_colored[:, w, h] = [255, 0, 0]  # 多分红色
                        if diff[w][h] < 0:
                            output_colored[:, w, h] = [255, 255, 0]  # 少分绿色
                        if diff[w][h] == 0 and gt[j].cpu().numpy().squeeze(0)[w][h] == 1:
                            output_colored[:, w, h] = [0, 255, 0]  # 少分绿色

                output_colored=Image.fromarray(np.uint8(output_colored.transpose(1,2,0)))
                output_colored.save(os.path.join(predict_epoch,'color'+data['name'][j].split('\\')[-1]))

                # 保存结果图像
            gt_flat = gt.cpu().numpy()[:,].flatten()
            pred_prob_flat = output[:,].cpu().numpy().flatten()
            Auc+= roc_auc_score(gt_flat, pred_prob_flat)


            gt=gt.to('cuda')
            confmat.update(gt.flatten(), torch.where(output>0.5,0,1).flatten())
            dice_loss += dice_criterion(output_,gt,weight=np.array([]))
            # dice_loss += dice_criterion(output_[:,0],gt)
        confmat.reduce_from_all_processes()
        # dice.reduce_from_all_processes()

    dice_loss/=(math.ceil(total_batch/args.val_batch_size))
    Auc/=(math.ceil(total_batch/args.val_batch_size))

    val_info, se, sp, acc = confmat.back()
    val_info += f'\nAUC:{Auc:.4f}\n' + 'val_dice:{:.4f}\n'.format(1 - dice_loss)
    print(val_info)


    # print(f"dice coefficient: {dice.value.item():.3f}")

    return 1-dice_loss,se,sp,acc,val_info

