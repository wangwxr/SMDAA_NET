import numpy as np
import torch


def cal_wight_map(y,y_hat,entropy_map,bs,mask =  0):

    weight_map = torch.ones_like(y).cpu().numpy().squeeze(1)
    short_image = ((y[:, 0].cpu().detach().numpy() - \
                    np.array(torch.sigmoid(y_hat).cpu().detach().numpy()[:, 0] > 0.5).astype(int)) > 0).astype(
        int)
    long_image = ((y[:, 0].cpu().detach().numpy() - \
                   np.array(torch.sigmoid(y_hat).cpu().detach().numpy()[:, 0] > 0.5).astype(int)) < 0).astype(
        int)
    right_image = ((y[:, 0].cpu().detach().numpy() - \
                    np.array(torch.sigmoid(y_hat).cpu().detach().numpy()[:, 0] > 0.5).astype(int)) == 0).astype(
        int)
    # assert ((right_image.sum(1).sum(1) / (right_image.shape[1] * right_image.shape[2]) + \
    #         long_image.sum(1).sum(1) / (long_image.shape[1] * long_image.shape[2]) + \
    #         short_image.sum(1).sum(1) / (short_image.shape[1] * short_image.shape[2])) \
    #         == np.ones(bs)).all()
    # TODO:这里的weight用1-（像素所占原图比例）来计算
    part_ratio = [right_image.sum(1).sum(1) / (right_image.shape[1] * right_image.shape[2]), \
                  long_image.sum(1).sum(1) / (long_image.shape[1] * long_image.shape[2]), \
                  short_image.sum(1).sum(1) / (short_image.shape[1] * short_image.shape[2])]
    smooth = 0.002
    alpha = 0.5
    beta = 0.8
    weight_map=weight_map*np.exp(np.sqrt(entropy_map*alpha))+ beta*np.log(1 + np.sqrt(short_image * (1/(part_ratio[2][:,np.newaxis,np.newaxis] + smooth )) ))
    assert not np.isnan(weight_map).any()
    return   short_image, long_image ,  weight_map

def cal_wight_map_ske(y,y_hat,bs):
    weight_map = torch.ones_like(y).cuda()
    short_ske = ((y - y_hat) > 0).long()
    long_ske = ((y - y_hat) < 0).long()
    right_ske = ((y - y_hat) == 0).long()
    # assert ((right_ske.sum(1).sum(1) / (right_ske.shape[1] * right_ske.shape[2]) + \
    #         long_ske.sum(1).sum(1) / (long_ske.shape[1] * long_ske.shape[2]) + \
    #         short_ske.sum(1).sum(1) / (short_ske.shape[1] * short_ske.shape[2]))  == torch.ones(bs).cuda()).all()
    part_ratio = [right_ske.sum(1).sum(1) / (right_ske.shape[1] * right_ske.shape[2]), \
                  long_ske.sum(1).sum(1) / (long_ske.shape[1] * long_ske.shape[2]), \
                  short_ske.sum(1).sum(1) / (short_ske.shape[1] * short_ske.shape[2])]
    alpha = 0.5
    beta = 1
    # weight_map=weight_map+ alpha * short_image * (1/part_ratio[2][:,np.newaxis,np.newaxis] )+ entropy_map


    # weight_map = weight_map  + beta * torch.log(
    #     1 + torch.sqrt(short_ske * (1 / part_ratio[2].unsqueeze(1).unsqueeze(1))))
    # weight_map = torch.where(torch.isnan(weight_map), torch.full_like(weight_map, 0), weight_map)

    return short_ske, long_ske, weight_map



def normliaztion(map):
    min_entropy = np.min(map, axis=(1, 2))
    min_entropy = min_entropy[:, np.newaxis, np.newaxis]
    max_entropy = np.max(map, axis=(1, 2))
    max_entropy = max_entropy[:, np.newaxis, np.newaxis]
    entropy_map = (map - min_entropy) / (max_entropy - min_entropy)
    return entropy_map