import cv2
import numpy
import numpy as np
import torch
from PIL import Image
import numpy as np
from PIL import Image
import os
from torch import nn

class SK_loss(nn.Module):
    def __init__(self):
        super(SK_loss, self).__init__()

    def forward(self,predict):
        assert torch.unique(predict).shape == 0
        sk_pre = skelton(predict)
        loss = 0
        return  loss

        # tensor
def skelton(img):
    img = img.clone()

    img = np.array(img.cpu()).astype('uint8')
    img = img.squeeze(1)
    final_img = numpy.asarray(img)
    x_tr = 0
    batch = img.shape[0]
    height = img.shape[1]
    width = img.shape[2]
    # 使用PIL打开GIF图像
    # im = Image.open(imgpath)
    # 转为RGB格式
    # im = im.convert('RGB')
    # 转换为OpenCV格式的numpy数组
    result = np.zeros((batch,height, width), dtype=np.uint8)

    for ii in img:
        # 高斯滤波
        # ii = cv2.GaussianBlur(ii, (5, 5), 1)
        # Otsu自动阈值化
        ret, thresh = cv2.threshold(ii, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 检查threshold结果
        if thresh is None:
            print('Error thresholding image')
            exit(1)
        # 骨架化
        size = np.size(ii)
        skell = np.zeros(ii.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        iii = 0
        while (not done):
            eroded = cv2.erode(thresh, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(thresh, temp)
            skell = cv2.bitwise_or(skell, temp)
            thresh = eroded.copy()
            zeros = size - cv2.countNonZero(thresh)
            if zeros == size:
                done = True
            iii+=1
            if iii>50:
                break
        result[x_tr] = skell


        # for ii in range(1, height - 1):
        #     for j in range(1, width - 1):
        #         if skell[ii, j] == 255:
        #             获取周围八个像素值
                    # neighborhood = skell[ii - 1:ii + 2, j - 1:j + 2]
                    # 检查是否周围八个像素都为0
                    # if np.sum(neighborhood) == 255:
                    #     result[x_tr][ii, j] = 0
                    # else:
                    #     result[x_tr][ii, j] = skell[ii, j]
                # else:
                #     result[x_tr][ii, j] = skell[ii, j]
        assert len(np.unique(skell)) <= 2

        x_tr+=1
        assert not np.array_equal(result,img)

        kernel = np.ones((3, 3), np.uint8)
        result = cv2.dilate(result, kernel)
        return result

        # cv2.imwrite('{:0>2d}.tif'.format(i), skell)
    # 显示结果
# cv2.imshow("skeleton", skell)
# cv2.imwrite(filename, img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == '__main__':
    path = r'D:\2021\wwww\dataset\DRIVE\training\1st_manual'
    datalist = os.listdir(path)
    list = [os.path.join(path,i) for i in datalist]
    for i in range(len(list) ):
        a = skelton(torch.tensor(Image.open(list[i])) )
        Image.fromarray(np.array(a)).show()
# 骨架化后修复断点

# 闭运算补连接断点
# kernel = np.ones((5,5),np.uint8)
# close = cv2.morphologyEx(skell, cv2.MORPH_CLOSE, kernel)
#
# # 距离变换连接
# dist = cv2.distanceTransform(skell, cv2.DIST_L2, 3)
#
# markers = np.zeros_like(skell)
# markers[dist<3] = 255
# segmentation = cv2.watershed(dist, markers)
# # 将单通道dist复制三份合成三通道
# dist_3ch = np.dstack([dist]*3)
#
# # 然后作为src传入watershed
# segmentation = cv2.watershed(dist_3ch, markers)
# seg_img = np.where(segmentation == -1, 255, 0).astype(np.uint8)
#
# # 显示修复结果
# cv2.imshow('close', close)
# cv2.imshow('watershed', seg_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()