import torch
import torch.nn.functional as F
import numpy

from abc import ABC
from torch import nn


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self,
                 temperature=0.07,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=100,
                 ignore_index=-100,
                 device='cuda:0',
                 memory =True ,
                 memory_size=100,
                 pixel_update_freq=10,
                 pixel_classes= 2 ,
                 dim =256
                 ):
        super(PixelContrastLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_label = ignore_index
        self.max_samples = max_samples
        self.max_views = max_views
        self.device = device
        self.memory = memory
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq

        if self.memory:
            self.segment_queue = torch.randn(pixel_classes, self.memory_size, dim)
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.segment_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
            self.pixel_queue = torch.zeros(pixel_classes, self.memory_size, dim)
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.pixel_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
    # def _anchor_sampling(self, X, y_hat, y , mask,entropy):
    def _anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        classes = []
        total_classes = 0
        # 正负样本都采样
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            #TODO:这两行要不要改呢，感觉没啥用啊
            classes.append(this_classes)
            total_classes += len(this_classes)
        if total_classes == 0:
            return None, None
        #                      1024//4
        n_view = self.max_samples // total_classes
        n_view = min(n_view,self.max_views)
        #这个max_view是什么玩意

        #TODO:这行感觉不对劲，所以我注释掉了
        # 这个n_view 取多少合适 需要同时控制hard 和 easy 一个n_view肯定不够  两个都取最小的吧
        # n_view_hard=min(min(short_num),min(long_num))
        # n_view_easy=min(min(right_0_num),min(right_1_num))
        #
        # n_view_easy=min(n_view_easy,self.max_views)
        # n_view_hard = min(n_view_hard, self.max_views)

        #先只加血管少分的 都加
        # X_ = torch.zeros((total_classes,n_view_hard + n_view_easy,feat_dim),dtype=torch.float).to(self.device)
        # y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(self.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)
        #
        X_ptr = 0

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            for cls_id in this_classes:
            #     short_indics = ()
            #this_classes为1时是少分的 为0是多分的
                #todo：这是把少分的和多分的作对比损失吧
                # 应该是把少分的和背景作对比损失 使其拉开距离 所以硬样本就是少分的

                    # hard_indices = ((this_y_hat == cls_id) & (this_y == (1 - cls_id))).nonzero()
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()

                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
                                  #实际为1                  #预测为0
                # hard_entropy_indics = hard_indices[torch.argsort(this_entropy[hard_indices].T,descending=True)].squeeze(2).T
                # easy_entropy_indics = easy_indices[torch.argsort(this_entropy[easy_indices].T,descending=True)].squeeze(2).T

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                # # 这个n_view不能太大
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]

                # hard_indices = hard_entropy_indics[:n_view_hard]
                # easy_indices = easy_indices[easy_entropy_indics[:num_easy_keep]]
                #TODO:这里应该用不着easy_indics吧
                # indices = hard_indices
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                # try :
                #     X_[X_ptr, :, :] = X[ii, indices.cpu(), :].squeeze(1)
                # except RuntimeError:
                #     print(1)
                X_[X_ptr, :, :] = X[ii, indices.cpu(), :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr+=1

                #这应该是随机取硬样本中的一部分开始对比学习
        return X_, y_

    def _sample_negative(self, Q):
        class_num, memory_size, feat_size = Q.shape

        x_ = torch.zeros((class_num * memory_size, feat_size)).float().to(self.device)
        y_ = torch.zeros((class_num * memory_size, 1)).float().to(self.device)

        sample_ptr = 0
        for c in range(class_num):
            # if c == 0:
            #     continue
            this_q = Q[c, :memory_size, :]
            x_[sample_ptr:sample_ptr + memory_size, ...] = this_q
            y_[sample_ptr:sample_ptr + memory_size, ...] = c
            sample_ptr += memory_size
        return x_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)  # (anchor_num ) × 1
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)  # (anchor_num × n_view) × feat_dim
        #锚点的特征
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        # (anchor_num × n_view) × (anchor_num × n_view)
        mask = torch.eq(y_anchor, y_contrast.T).float().to(self.device)
        # (anchor_num × n_view) × (anchor_num × n_view)  #下面实质就是每行和每列做点积
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        #这个logits_max为什么全为一个值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  #实质是标准化
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        #下面这是为了把对角线置零
        mask_ones = torch.ones_like(mask)
        k = min(mask_ones.shape[0],mask_ones.shape[1])
        logits_mask = mask_ones.scatter_(1, torch.arange(k).view(-1, 1).to(self.device), 0)

        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        #这等于0因为log（e^b）=b     一点负样本没有学个屁 回去加上
        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
    def _dequeue_and_enqueue(self, keys, labels):

        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]
        labels = torch.nn.functional.interpolate(labels, (keys.shape[2], keys.shape[3]), mode='nearest')


        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1).cuda()
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x >= 0]
            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                lb = int(lb.item())
                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)  #取10个放进队列里
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size
    # def forward(self, feats_1, feats_2, short_img, labels=None, predict=None):
    def forward(self, feats, labels=None, predict=None, mask = None,encropy_map = None,queue=None):
        queue = self.pixel_queue  if self.memory else None #2,100,256
        queue_feats = feats.detach()

        # queue会被随机初始化 2,100,256
        labels = labels.float().clone()  #2,1,512,512
        origin_labels = labels.float().clone()
        predict = predict.float().clone() #2,1,512,512
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = torch.nn.functional.interpolate(predict, (feats.shape[2], feats.shape[3]), mode='nearest')

        labels = labels.long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        # feats: N×(HW)×C
        # labels: N×(HW)
        # predict: N×(HW)
        feats_, labels_ = self._anchor_sampling(feats, labels, predict )
        loss = self._contrastive(feats_, labels_, queue=queue)
        #todo:这个detach要去掉吗
        self._dequeue_and_enqueue(queue_feats.detach(), origin_labels.detach())

        return loss
class EmbeddingHead(nn.Module):
    #TODO:这块要改吗？？？

    def __init__(self, dim_in=256, embed_dim=256, embed='convmlp'):
        super(EmbeddingHead, self).__init__()

        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)
            )

    def forward(self, x):
            return F.normalize(self.embed(x), p=2, dim=1)