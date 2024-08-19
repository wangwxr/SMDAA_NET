from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mcdaunet import *
BN_EPS = 1e-4

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, t=3):
        super(RecurrentConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.t = t

    def forward(self, x):
        for _ in range(self.t):
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
        return x
class dual_path_doubleconv(nn.Module):  #只加在编码器里
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(dual_path_doubleconv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.cr_conv1 = RecurrentConvLayer(in_channels,mid_channels)
        self.cr_conv2 = RecurrentConvLayer(mid_channels,out_channels)
        self._1conv = nn.Conv2d(out_channels*2,out_channels,kernel_size=1)
        self._1relu = nn.ReLU(inplace=True)
        self._1xselfconv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        x_path1 = self.conv2(self.conv1(x))
        x_path2 = self.cr_conv2(self.cr_conv1(x))
        final_x = torch.concat(x_path1,x_path2)
        final_x = self._1relu(self._1conv(final_x))
        final_x = final_x + self._1xselfconv(x)
        #这块完了要加注意力吗？
        return final_x




class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            # DoubleConv(in_channels, out_channels)
            DoubleConv(in_channels, out_channels)

        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class EmbeddingHead(nn.Module):
    #TODO:这块要改吗？？？

    def __init__(self, dim_in=512, embed_dim=512, embed='convmlp'):
        super(EmbeddingHead, self).__init__()

        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)
            )

    def forward(self, x):
            return F.normalize(self.embed(x), p=2, dim=1)




class SMDAA(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32,
                 emb : bool = False,


                ):
        super(SMDAA, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.bn = True
        self.BatchNorm = False
        # self.in_conv =DoubleConv(in_channels, base_c)
        # self.in_conv = dual_path_doubleconv(in_channels, base_c)
        self.in_conv = mulite_scale_conv(3, 32, kernel_size=3, bn=True, BatchNorm=False, res= True)



        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 )

        # 添加密集连接块
        self.dense_block1 = self._make_dense_block(base_c, base_c)

        self.dense_block2 = self._make_dense_block(base_c*2, base_c*2)
        self.dense_block3 = self._make_dense_block(base_c*2 * 2, base_c*2 * 2)
        self.dense_block4 = self._make_dense_block(base_c*2 * 4, base_c*2 * 4)
        self.dense_block5= self._make_dense_block(base_c * 16, base_c * 16)

        self.up1 = Up(1248, base_c * 8 , bilinear)
        self.up2 = Up(608, base_c * 4 , bilinear)
        self.up3 = Up(288, base_c * 2 , bilinear)
        self.up4 = Up(96, base_c, bilinear)

        # self.out_conv = OutConv(base_c, num_classes)
        self.out_conv = OutConv(320, num_classes)
        # self.out_conv = OutConv(320, num_classes)

        self.emb = emb
        # self.out_layer = OutLayer(320, 2)    #X1+X2+X3+X4+X5 ,the channels becomes 320


        if self.emb:
            self.embed_head = EmbeddingHead()
        self.se = SELayer(320)
            # self.se = coord_attention(320)

        self.side_5 = nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1,
                                bias=True)  # The X1, X2,X3,X4 and X5 obtained before fusion of each layer in the MLF module
        self.side_6 = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_9 = nn.Conv2d(32, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.coord1 = coord_attention(32)
        self.coord2 = coord_attention(160)
        self.coord3 = coord_attention(352)
        self.coord4 = coord_attention(736)
        self.coord5 = coord_layer(320)
    def _make_dense_block(self, in_channels, out_channels):
        layers = [DoubleConv(in_channels, out_channels)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor,  feat=False) -> Dict[str, torch.Tensor]:
        _, _, img_shape, _ = x.size()
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 应用密集连接块
        x1_dense = self.dense_block1(x1)
        x2_dense = self.dense_block2(x2)
        x3_dense = self.dense_block3(x3)
        x4_dense = self.dense_block4(x4)
        x5_dense = self.dense_block5(x5)

        # 连接密集连接块的输出
        x2_ds = torch.cat([x2, nn.MaxPool2d(kernel_size=2)(x1),x2_dense], dim=1)
        x3_ds = torch.cat([x3,nn.MaxPool2d(kernel_size=2)((nn.MaxPool2d(kernel_size=2)(x1))), nn.MaxPool2d(kernel_size=2)(x2), x3_dense], dim=1)
        x4_ds = torch.cat([x4,nn.MaxPool2d(kernel_size=2)(nn.MaxPool2d(kernel_size=2)((nn.MaxPool2d(kernel_size=2)(x1)))), nn.MaxPool2d(kernel_size=2)((nn.MaxPool2d(kernel_size=2)(x2))), nn.MaxPool2d(kernel_size=2)(x3), x4_dense], dim=1)
        x5_ds = torch.cat([x5, nn.MaxPool2d(kernel_size=2)(nn.MaxPool2d(kernel_size=2)(nn.MaxPool2d(kernel_size=2)((nn.MaxPool2d(kernel_size=2)(x1))))),nn.MaxPool2d(kernel_size=2)(nn.MaxPool2d(kernel_size=2)(nn.MaxPool2d(kernel_size=2)(x2))), nn.MaxPool2d(kernel_size=2)(nn.MaxPool2d(kernel_size=2)(x3)), nn.MaxPool2d(kernel_size=2)(x4), x5_dense], dim=1)

        #coord_attention
        x1    = self.coord1(x1)
        x2_ds = self.coord2(x2_ds)
        x3_ds = self.coord3(x3_ds)
        x4_ds = self.coord4(x4_ds)


        x6 = self.up1(x5, x4_ds)
        x7 = self.up2(x6, x3_ds)
        x8 = self.up3(x7, x2_ds)
        x9 = self.up4(x8, x1)

        side_5 = F.interpolate(x6, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_6 = F.interpolate(x7, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_7 = F.interpolate(x8, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_8 = F.interpolate(x9, size=(img_shape, img_shape), mode='bilinear', align_corners=True)
        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)
        side_9 = self.side_9(x1)


        # x8 = self.up3(x7, x2_ds)
        # x9 = self.up4(x8, x1)
        # out = self.se(side_5, side_6, side_7, side_8, side_9)
        #改成coordattention试试
        out = self.coord5(side_5, side_6, side_7, side_8, side_9)

        # logits = self.out_conv(out)
        logits = self.out_conv(out)
        if feat:
            if self.emb:
                return x5, logits, self.embed_head(x5)
            else:
                return x5, logits
        else:
            return logits


class senet(nn.Module):
    def __init__(self,inchannels,ratio = 16):
        super(senet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Sequential(
            nn.Linear(inchannels,inchannels//16,False),
            nn.ReLU(),
            nn.Linear(inchannels//16,inchannels,False),
            nn.Sigmoid(),
        )

    def forward(self,x):
        b,c,h,w = x.size()
        avg = self.avgpool(x).view([b,c])
        fc = self.fc(avg).view([b,c,1,1])

        return x*fc

class OutLayer(nn.Module):
    """Out:  n_channels from 320 to 2"""
    def __init__(self, input_channels, output_channels):
        super(OutLayer, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//5, kernel_size=1, padding=0),
            nn.BatchNorm2d(input_channels//5),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels//5, input_channels // 5, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels // 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//5, output_channels, kernel_size=1, padding=0),
           )

    def forward(self, x):
        x = self.fc(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)

class SELayer(nn.Module):
    """[X1,X2,X3,X4,X5]-->SE"""
    def __init__(self, channel, reduction=16,stride=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1,x2,x3,x4,x5):
        x = torch.cat([x1, x2,x3,x4,x5], dim=1)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        b, c, _, _ = out.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out += residual
        out = self.relu(out)
        return out
class coord_attention(nn.Module):
    def __init__(self,channel,reduction=16):
        super(coord_attention, self).__init__()
        self._1x1conv = nn.Conv2d(channel,channel//16,kernel_size=1,stride=1,bias=False)
        self.bn       = nn.BatchNorm2d(channel//reduction)
        self.relu     = nn.ReLU()
        self.swish = Swish()
        self.F_h      = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
        self.F_w      = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
    def forward(self,x):
        residual = x
        #b,c,h,w
        _,_,h,w = x.size()

        #b,c,h,w->b,c,h,1->b,c,1,h
        x_h = torch.mean(x,dim=3,keepdim=True).permute(0,1,3,2)

        #b,c,h,w->b,c,1,w
        x_w = torch.mean(x,dim=2,keepdim=True)
        x_cat_bn_relu = self.relu(self.bn(self._1x1conv(torch.cat((x_h,x_w),3))))

        x_cat_split_h,x_cat_split_w = x_cat_bn_relu.split([h,w],3)

        s_h =  self.sigmoid_h(self.F_h(x_cat_split_h.permute(0,1,3,2)))
        s_w =  self.sigmoid_w(self.F_w(x_cat_split_w))

        out = x*s_h.expand_as(x)*s_w.expand_as(x)
        out += residual
        out = self.swish(out)
        return out


#没用到呢这个

class coord_layer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(coord_layer, self).__init__()
        self._1x1conv = nn.Conv2d(channel,channel//16,kernel_size=1,stride=1,bias=False)
        self.bn       = nn.BatchNorm2d(channel//reduction)
        self.relu     = nn.ReLU()
        self.F_h      = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
        self.F_w      = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
    def forward(self,x1,x2,x3,x4,x5):
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        residual = x
        #b,c,h,w
        _,_,h,w = x.size()

        #b,c,h,w->b,c,h,1->b,c,1,h
        x_h = torch.mean(x,dim=3,keepdim=True).permute(0,1,3,2)

        #b,c,h,w->b,c,1,w
        x_w = torch.mean(x,dim=2,keepdim=True)
        x_cat_bn_relu = self.relu(self.bn(self._1x1conv(torch.cat((x_h,x_w),3))))

        x_cat_split_h,x_cat_split_w = x_cat_bn_relu.split([h,w],3)

        s_h =  self.sigmoid_h(self.F_h(x_cat_split_h.permute(0,1,3,2)))
        s_w =  self.sigmoid_w(self.F_w(x_cat_split_w))

        out = x*s_h.expand_as(x)*s_w.expand_as(x)
        out += residual
        out = self.relu(out)
        return out


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 创建Swish激活函数
swish = Swish()