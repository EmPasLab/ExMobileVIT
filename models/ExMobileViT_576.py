from models.basic_block import MobileViTBlock, MV2Block
from models.basic_block import conv_nxn_bn, conv_1x1_bn
import torch
import torch.nn as nn
import pytorch_lightning as pl

#################
# 3  -  4  -  5 #
#    Multiple   #
#1/3 - 1/2 -  3 #
#===============#
# 1  -  2  -  15#
#################

class ExMobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2_1 = conv_1x1_bn(channels[-6], 32)
        self.conv2_2 = conv_1x1_bn(channels[-4], 64)
        self.conv2   = conv_1x1_bn(channels[-2], 480)

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.pool25 = nn.AvgPool2d(25)
        self.pool9 = nn.AvgPool2d(9)
        self.fc = nn.Linear(576, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)

        x = self.mv2[4](x)
        x = self.mvit[0](x)
        
        x1 = self.conv2_1(x)
        x1 = self.pool25(x1).view(-1, x1.shape[1])

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x2 = self.conv2_2(x)
        x2 = self.pool9(x2).view(-1, x2.shape[1])

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        
        x = self.conv2(x)

        x3 = self.pool(x).view(-1, x.shape[1])
        x_cat = torch.cat([x1, x2, x3], dim=1)
        
        x = self.fc(x_cat)
        
        return x
