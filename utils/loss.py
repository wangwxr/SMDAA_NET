# -*- encoding: utf-8 -*-
#Time        :2020/12/19 21:17:50
#Author      :Chen
#FileName    :loss.py
#Version     :1.0
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
#iou loss
def iou_loss(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return 1 - iou_score



"""Dice loss"""



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()


    def forward(self, pred, target,tiny=np.array([]),huge=np.array([]),weight=[],is_sigmoid = True):
        # short[1,0,0]、long[0,1,0]
        tiny=torch.from_numpy(tiny)
        huge=torch.from_numpy(huge)
        weight=torch.from_numpy(weight)
        smooth = 1
        dice_score=0
        dice_loss=0
        size = pred.size(0)
        if is_sigmoid:
            pred=torch.sigmoid(pred)
        pred_flat_begin = pred.view(size, -1)
        target_flat_begin = target.view(size, -1)
        if weight.sum()!=torch.tensor(-1):
            for i in range(pred.shape[0]):
                # short_weight_map = (torch.eq(tiny[i], 0).long()) * weight[i][0] + (torch.eq(tiny[i], 1).long()) * weight[i][2]
                # # long_weight_map = (torch.eq(huge[i], 0).long()) * weight[i][0] + (torch.eq(huge[i], 1).long()) * weight[i][1]
                # # weight_map = (short_weight_map/weight[i][0]) * (long_weight_map/weight[i][0])
                # weight_map = (short_weight_map/weight[i][0])
                weight_map = weight[i]
                # weight_map=short_weight_map
                weight_map_flat = weight_map.view(-1).cuda()
                pred_flat=pred_flat_begin[i]
                target_flat = target_flat_begin[i]
                weight_flat = weight_map_flat
                assert 255 not in target_flat
                intersection = pred_flat * target_flat
                dice_score__ = (2 * (intersection*weight_flat).sum() + smooth)/((pred_flat*weight_flat).sum() + (target_flat*weight_flat).sum() + smooth)
                dice_loss__ = 1 - dice_score__.sum()
                dice_loss += dice_loss__
        else:
            for i in range(pred.shape[0]):
                pred_flat=pred_flat_begin[i]
                target_flat=target_flat_begin[i]
                assert 255 not in target_flat
                intersection = pred_flat * target_flat
                dice_score__ = (2 * (intersection).sum() + smooth)/((pred_flat).sum() + (target_flat).sum() + smooth)
                dice_loss__ = 1 - dice_score__.sum()
                dice_loss += dice_loss__
        dice_loss=dice_loss/size
        assert dice_loss>=0

        return dice_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

"""BCE + DICE Loss"""





""" Entropy Minimization"""
class softCrossEntropy(nn.Module):
    def __init__(self, ignore_index= -1):
        super(softCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss
        """
        assert inputs.size() == target.size()
        mask = (target != self.ignore_index)

        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.mean(torch.mul(-log_likelihood, target)[mask])

        return loss


"""Maxsquare Loss"""
class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1):
        super().__init__()
        self.ignore_index = ignore_index
        #self.num_class = num_class
    
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        #label = (prob != self.ignore_index)
        loss = -torch.mean(torch.pow(prob, 2) + torch.pow(1-prob, 2)) / 2
        return loss