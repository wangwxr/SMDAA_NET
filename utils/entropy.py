import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1e-7

def sig_label_to_hard(sig_pls, pseudo_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    pseudo_label = pseudo_label_s.float()

    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2

    return pseudo_labels.unsqueeze(dim=1)

def sig_label_to_hard_en_map(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    pseudo_label = pseudo_label_s.float()

    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    en_map = prediction_entropy.clone()


    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    high_pseudo_label = high_entropy.int()
    high_pseudo_label = high_pseudo_label.float()

    high_pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        high_pseudo_labels = high_pseudo_labels.cuda()
    high_pseudo_labels[high_pseudo_label[:, 0] == 1] = 1
    high_pseudo_labels[high_pseudo_label[:, 1] == 1] = 2
    high_pseudo_labels = high_pseudo_labels.unsqueeze(dim=1)


    return pseudo_labels.unsqueeze(dim=1), en_map, high_pseudo_labels

def sig_label_to_hard_ensure(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    low_entropy = prediction_entropy.clone()
    low_entropy_clo = prediction_entropy.clone()
    low_entropy[low_entropy_clo > en_label_threshold] = 0
    low_entropy[low_entropy_clo <= en_label_threshold] = 1

    pseudo_label = pseudo_label_s.int() & low_entropy.int()
    pseudo_label = pseudo_label.float()


    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2
    pseudo_labels = pseudo_labels.unsqueeze(dim=1)

    ignores = torch.ones([sig_pls.size()[0], sig_pls.size()[1], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        ignores = ignores.cuda()
    ignores[high_entropy == 1] = 0

    return pseudo_labels, ignores

##todo：用这个应该是
def sig_label_to_hard_en(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1     #用熵值生成伪标签
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()


    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels = pseudo_labels.unsqueeze(dim=1)

    ignores = torch.zeros([sig_pls.size()[0], sig_pls.size()[1], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        ignores = ignores.cuda()
    ignores[high_entropy == 1] = 1

    # batch级别
    entropy_map = prediction_entropy.clone()
    weight_en = 1 - ((entropy_map*ignores).sum(dim=[0, 2, 3]) / (ignores.sum(dim=[0, 1, 2, 3])))
    # weight_en = 1 - ((entropy_map).sum(dim=[0, 2, 3]) / (entropy_map.shape[0]*entropy_map.shape[2]*entropy_map.shape[3]))
    weight_class = 1 - ((pseudo_label*ignores).sum(dim=[0, 2, 3]) / (ignores.sum(dim=[0, 1, 2, 3])))
    # weight_class = 1 - ((pseudo_label).sum(dim=[0, 2, 3]) / (entropy_map.shape[0]*entropy_map.shape[2]*entropy_map.shape[3]))
    weight = (weight_en + weight_class) / 2

    return pseudo_labels, ignores, weight

def sig_label_to_hard_en_ignore(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()
    # 去除简单样本
    entropy = prediction_entropy.clone()
    pseudo_label[entropy < en_label_threshold] = -1

    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()

    pseudo_labels[pseudo_label[:, 0] == -1] = -1
    pseudo_labels[pseudo_label[:, 1] == -1] = -1

    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2

    return pseudo_labels.unsqueeze(dim=1)

def sig_label_to_hard_en_ignore_1(sig_pls, pseudo_label_threshold, en_label_threshold):
    # 伪标签
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()

    # # 背景概率
    # bak_label_s = sig_pls.clone()
    # bak_label_s_mean = torch.mean(1 - bak_label_s, dim=[1])
    # # 背景熵
    # bak_prediction_entropy = sigmoid_entropy(bak_label_s_mean)
    # bak_prediction_entropy[torch.isnan(bak_prediction_entropy)] = 0
    # bak_prediction_entropy = ((bak_prediction_entropy - bak_prediction_entropy.min()) * (
    #         1 / (bak_prediction_entropy.max() - bak_prediction_entropy.min())))
    # for cls in range(sig_pls.size()[1]):
    #     bak_prediction_entropy[pseudo_label[:, cls] == 1] = 0


    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()

    # 去除简单样本
    entropy = prediction_entropy.clone()
    # pseudo_label[entropy < en_label_threshold] = -1
    pseudo_labels[(entropy[:, 0] < en_label_threshold) & (pseudo_label[:, 0] == 1)] = -1
    pseudo_labels[(entropy[:, 0] > 0.8) & (pseudo_label[:, 0] == 1)] = -1 # 删除
    pseudo_labels[(entropy[:, 1] < en_label_threshold) & (pseudo_label[:, 1] == 1)] = -1
    pseudo_labels[(entropy[:, 1] < 0.8) & (pseudo_label[:, 1] == 1)] = -1 # 删除

    # pseudo_labels[pseudo_label[:, 0] == -1] = -1
    # pseudo_labels[pseudo_label[:, 1] == -1] = -1

    pseudo_labels[(entropy[:, 0] > en_label_threshold) & (entropy[:, 0] < 0.8) & (pseudo_label[:, 0] == 1)] = 1
    # pseudo_labels[(entropy[:, 0] > en_label_threshold) & (pseudo_label[:, 0] == 1)] = 1
    pseudo_labels[(entropy[:, 1] > en_label_threshold) & (entropy[:, 0] < 0.8) & (pseudo_label[:, 1] == 1)] = 2
    # pseudo_labels[(entropy[:, 1] > en_label_threshold) & (pseudo_label[:, 1] == 1)] = 2

    return pseudo_labels.unsqueeze(dim=1)

def sig_label_to_hard_en_high(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() & high_entropy.int()
    pseudo_label = pseudo_label.float()


    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2

    return pseudo_labels.unsqueeze(dim=1)

def sig_label_to_hard_en_low(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 0
    high_entropy[high_entropy <= en_label_threshold] = 1

    pseudo_label = pseudo_label_s.int() & high_entropy.int()
    pseudo_label = pseudo_label.float()


    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2

    return pseudo_labels.unsqueeze(dim=1)

def sig_label_to_hard_en_weight(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()
    # 图片级别
    weight_en = 1 - (high_entropy.sum(dim=[2, 3]) / (pseudo_label.sum(dim=[2, 3]) + EPS))
    weight_class = 1 - (pseudo_label.sum(dim=[2, 3]) / (sig_pls.size()[2] * sig_pls.size()[3]))
    weight = (weight_en + weight_class)/2
    # batch级别
    # weigt_en = 1 - (high_entropy.sum(dim=[0, 2, 3]) / pseudo_label.sum(dim=[0, 2, 3]))
    # weigt_class = 1 - (pseudo_label.sum(dim=[0, 2, 3]) / (sig_pls.size()[2] * sig_pls.size()[3]))
    # weigt = (weigt_en + weigt_class) / 2
    # print(weight)
    return weight
def sig_label_to_hard_en_weight_1(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()
    # 图片级别
    weight_en = (high_entropy.sum(dim=[2, 3]) / (pseudo_label.sum(dim=[2, 3]) + EPS))
    weight_class = 1 - (pseudo_label.sum(dim=[2, 3]) / (sig_pls.size()[2] * sig_pls.size()[3]))
    weight = (weight_en + weight_class)/2
    # print(weight)
    return weight

def sig_label_to_hard_en_weight_batch(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()

    # batch级别
    weight_en = 1 - (high_entropy.sum(dim=[0, 2, 3]) / (pseudo_label.sum(dim=[0, 2, 3]) + EPS))
    weight_class = 1 - (pseudo_label.sum(dim=[0, 2, 3]) / (sig_pls.size()[2] * sig_pls.size()[3]))
    weight = (weight_en + weight_class) / 2
    # print(weight)
    return weight

def sig_label_to_hard_PCL(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    # pseudo_label = high_entropy.int()
    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()


    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2
    pseudo_labels = pseudo_labels.unsqueeze(dim=1)
    # 确定样本
    low_entropy = prediction_entropy.clone()
    low_entropy_clo = prediction_entropy.clone()
    low_entropy[low_entropy_clo > en_label_threshold] = 0
    low_entropy[low_entropy_clo <= en_label_threshold] = 1
    pseudo_label_low = pseudo_label_s.int() & low_entropy.int()
    pseudo_label_low = pseudo_label_low.float()
    pseudo_labels_low = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels_low = pseudo_labels_low.cuda()
    pseudo_labels_low[pseudo_label_low[:, 0] == 1] = 1
    pseudo_labels_low[pseudo_label_low[:, 1] == 1] = 2
    pseudo_labels_low = pseudo_labels_low.unsqueeze(dim=1)

    en_map = prediction_entropy.clone()
    return pseudo_labels, en_map, pseudo_labels_low

def sig_label_to_hard_PCL_beifen(sig_pls, pseudo_label_threshold, en_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    prediction_entropy = sig_pls.clone()

    prediction_entropy = sigmoid_entropy(prediction_entropy)
    prediction_entropy[torch.isnan(prediction_entropy)] = 0
    prediction_entropy = ((prediction_entropy - prediction_entropy.min()) * (
            1 / (prediction_entropy.max() - prediction_entropy.min())))
    high_entropy = prediction_entropy.clone()
    high_entropy[high_entropy > en_label_threshold] = 1
    high_entropy[high_entropy <= en_label_threshold] = 0

    pseudo_label = pseudo_label_s.int() | high_entropy.int()
    pseudo_label = pseudo_label.float()


    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2
    pseudo_labels = pseudo_labels.unsqueeze(dim=1)
    # 确定样本
    low_entropy = prediction_entropy.clone()
    low_entropy_clo = prediction_entropy.clone()
    low_entropy[low_entropy_clo > en_label_threshold] = 0
    low_entropy[low_entropy_clo <= en_label_threshold] = 1
    pseudo_label_low = pseudo_label_s.int() & low_entropy.int()
    pseudo_label_low = pseudo_label_low.float()
    pseudo_labels_low = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels_low = pseudo_labels_low.cuda()
    pseudo_labels_low[pseudo_label_low[:, 0] == 1] = 1
    pseudo_labels_low[pseudo_label_low[:, 1] == 1] = 2
    pseudo_labels_low = pseudo_labels_low.unsqueeze(dim=1)

    # ignores = torch.ones([sig_pls.size()[0], sig_pls.size()[1], sig_pls.size()[2], sig_pls.size()[3]])
    # if torch.cuda.is_available():
    #     ignores = ignores.cuda()
    # ignores[high_entropy == 1] = 0
    # 背景熵计算
    en_map = prediction_entropy.clone()
    return pseudo_labels, en_map, pseudo_labels_low
def cal_entropy(predict_sigmoid):
    entropy =  sigmoid_entropy(predict_sigmoid)
    min_entropy = np.min(entropy,axis=(1,2))
    min_entropy = min_entropy[:,np.newaxis,np.newaxis]
    max_entropy = np.max(entropy,axis=(1,2))
    max_entropy = max_entropy[:,np.newaxis,np.newaxis]
    entropy_map = (entropy - min_entropy) /(max_entropy-min_entropy)
    return entropy_map


def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x * np.log(x + 1e-30) + (1 - x) * np.log(1 - x + 1e-30))
    #
    # return - (x * np.log2(x) + (1 - x) * np.log2(1 - x))
