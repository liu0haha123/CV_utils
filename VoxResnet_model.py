import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class VoxRes(nn.Module):
    def __init__(self,in_channels):
        super(VoxRes, self).__init__()
        self.Vox = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self,input):
        return self.Vox(input)+input

class VoxResNet(nn.Module):
    """基类"""
    def __init__(self,in_channels,num_classes,ftrlen =[32,64,64,64]):
        super(VoxResNet, self).__init__()
        ftr1,ftr2,ftr3,ftr4 = ftrlen

        # stage 1
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels, ftr1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ftr1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr1, ftr1, kernel_size=3, padding=1, bias=False)
            )
        # stage 2
        self.conv1_2 = nn.Sequential(
            nn.BatchNorm3d(ftr1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr1, ftr2, kernel_size=3, stride=(1,2,2), padding=1, bias=True)
            )
        self.voxres2 = VoxRes(ftr2)
        self.voxres3 = VoxRes(ftr2)
        # stage 3
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(ftr2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr2, ftr3, kernel_size=3, stride=(1,2,2), padding=1, bias=True)
            )
        self.voxres5 = VoxRes(ftr3)
        self.voxres6 = VoxRes(ftr3)
        # stage 4
        self.conv7 = nn.Sequential(
            nn.BatchNorm3d(ftr3),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr3, ftr4, kernel_size=3, stride=(1,2,2), padding=1, bias=True)
            )

        self.voxres8 = VoxRes(ftr4)
        self.voxres9 = VoxRes(ftr4)

    def foward_stage1(self,input):
        h = self.conv1_1(input)
        return  h

    def foward_stage2(self,input):
        h = self.conv1_2(input)
        h = self.voxres2(h)
        h = self.voxres3(h)
        return h

    def foward_stage3(self,input):
        h = self.conv4(input)
        h = self.voxres5(h)
        h = self.voxres6(h)

        return  h

    def foward_stage4(self,input):
        h = self.conv7(input)
        h = self.voxres8(h)
        h = self.voxres9(h)
        return h


    def forward(self,input):
        h1 = self.foward_stage1(input)
        h2 = self.foward_stage2(h1)
        h3 = self.foward_stage3(h2)
        h4 = self.foward_stage4(h3)
        return h4


class VoxResNet_C(VoxResNet):
    def __init__(self,in_channels,num_classes):
        super(VoxResNet_C, self).__init__(in_channels,num_classes)

        self.C1 = nn.Conv3d(32,num_classes,kernel_size=1,bias=True)
        self.C2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1, bias=True)
            )

        self.C3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1, bias=True)
            )
        self.C4 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1, bias=True)
            )
    def forward(self, input):
        h1 = self.foward_stage1(input)
        h2 = self.foward_stage2(h1)
        h3 = self.foward_stage3(h2)
        h4 = self.foward_stage4(h3)

        c1 = self.C1(h1)
        c2 = self.C2(h2)
        c3 = self.C3(h3)
        c4 = self.C4(h4)

        return c1+c2+c3+c4


if __name__ == '__main__':
    net = VoxResNet_C(87, 2)
    net.cuda()
    while True:
        data = torch.rand(1, 87, 5, 192, 192)
        c = net(Variable(data).cuda())
        print(c)