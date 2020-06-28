import torch
import torch.nn as nn


class input_Block(nn.Module):
    def __init__(self,input_channel):
        super(input_Block, self).__init__()
        self.bnRelu6 = nn.Sequential(
            nn.BatchNorm3d(input_channel),
            nn.ReLU6(inplace=True)
        )
        self.conv3d = nn.Conv3d(in_channels = input_channel,out_channels=input_channel,kernel_size=3,stride=1,padding=1)

    def forward(self,input):
        out = self.conv3d(self.bnRelu6(input))
        out = self.conv3d(self.bnRelu6(input))
        out = out+input
        return out


class downsample_Resdiual_Block(nn.Module):
    def __init__(self,input_channel):
        super(downsample_Resdiual_Block, self).__init__()
        self.bnRelu6_1 = nn.Sequential(nn.BatchNorm3d(input_channel),nn.ReLU6(inplace=True))
        self.bnRelu6_2 = nn.Sequential(nn.BatchNorm3d(2*input_channel), nn.ReLU6(inplace=True))
        self.conv3dC3S1 =nn.Conv3d(in_channels = 2*input_channel,out_channels=2*input_channel,kernel_size=3,stride=1,padding=1)
        self.conv3dC3S2 = nn.Conv3d(in_channels = input_channel,out_channels=2*input_channel,kernel_size=3,stride=2,padding=1)
        self.conv3dC1S2 = nn.Conv3d(in_channels = input_channel,out_channels=2*input_channel,kernel_size=1,stride=2,padding=0)


    def forward(self,input):
       res = self.conv3dC1S2(input)
       out = self.conv3dC3S2(self.bnRelu6_1(input))
       out = self.conv3dC3S1(self.bnRelu6_2(out))

       out = out+res

       return  out



if __name__ == '__main__':
    input_test= torch.autograd.Variable(torch.zeros((1,32,144,144,144)))
    model = downsample_Resdiual_Block(32)

    output = model(input_test)
    print(output.shape)
