import torch
import torch.nn as nn
import torch.nn.functional as F
from GridAttention2D import  GridAttention2D
import torch
import torch.nn as nn


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False):
    nonlinearity = nn.LeakyReLU(inplace=True)

    if batchnorm:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nonlinearity)
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nonlinearity)
    return layer


def deconv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    nonlinearity = nn.LeakyReLU(inplace=True)

    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=bias),
        nonlinearity)
    return layer

    
def downsample(in_channels,out_channels):
    #1x1卷积制作横等映射
    downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1)
    )
    return downsample

class RUNet_Pytorch_DeepSup(torch.nn.Module):
    #加入残差连接的unet 效果未知
    def __init__(self, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(RUNet_Pytorch_DeepSup, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.use_dropout = dropout

        self.contr_1_1 = conv2d(n_input_channels, n_filt)
        self.indentity1 = downsample(n_input_channels,n_filt)
        self.contr_1_2 = conv2d(n_filt, n_filt)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.contr_2_1 = conv2d(n_filt, n_filt * 2)
        self.indentity2 = downsample(n_filt, n_filt * 2)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4)
        self.indentity3 = downsample(n_filt*2, n_filt * 4)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8)
        self.indentity4 = downsample(n_filt * 4, n_filt * 8)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 16)
        self.deconv_1 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)
        # self.deconv_1 = nn.Upsample(scale_factor=2)  # does only upscale width and height; Similar results to deconv2d

        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)
        # self.deconv_2 = nn.Upsample(scale_factor=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4, stride=1)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)
        # self.deconv_3 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_2 = nn.Conv2d(n_filt * 4 + n_filt * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # 'nearest' a bit faster but results a little worse (~0.4 dice points worse)
        self.output_2_up = nn.Upsample(scale_factor=2, mode=upsample)

        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2, stride=1)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)
        # self.deconv_4 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_3 = nn.Conv2d(n_filt * 2 + n_filt * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_3_up = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height

        self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1)

        # no activation function, because is in LossFunction (...WithLogits)
        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inpt):
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_1_c = self.indentity1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        contr_1_2 = contr_1_2+contr_1_1_c
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_1_c = self.indentity2(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        contr_2_2 = contr_2_2+ contr_2_1_c
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_1_c = self.indentity3(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        contr_3_2 = contr_3_2+contr_3_1_c
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1(pool_3)
        contr_4_1_c = self.indentity4(pool_3)
        contr_4_2 = self.contr_4_2(contr_4_1)
        contr_4_2 = contr_4_2+contr_4_1_c
        pool_4 = self.pool_4(contr_4_2)

        if self.use_dropout:
            pool_4 = self.dropout(pool_4)

        encode_1 = self.encode_1(pool_4)
        encode_2 = self.encode_2(encode_1)
        deconv_1 = self.deconv_1(encode_2)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1(concat1)
        expand_1_2 = self.expand_1_2(expand_1_1)
        deconv_2 = self.deconv_2(expand_1_2)

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        deconv_3 = self.deconv_3(expand_2_2)

        # Deep Supervision
        output_2 = self.output_2(concat2)
        output_2_up = self.output_2_up(output_2)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1(concat3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        deconv_4 = self.deconv_4(expand_3_2)

        # Deep Supervision
        output_3 = output_2_up + self.output_3(concat3)
        output_3_up = self.output_3_up(output_3)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(concat4)
        expand_4_2 = self.expand_4_2(expand_4_1)

        conv_5 = self.conv_5(expand_4_2)

        final = output_3_up + conv_5

        return final

class AttUnet(nn.Module):
    def __init__(self,n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(AttUnet, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.use_dropout = dropout

        self.contr_1_1 = conv2d(n_input_channels, n_filt)
        self.indentity1 = downsample(n_input_channels, n_filt)
        self.contr_1_2 = conv2d(n_filt, n_filt)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.contr_2_1 = conv2d(n_filt, n_filt * 2)
        self.indentity2 = downsample(n_filt, n_filt * 2)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4)
        self.indentity3 = downsample(n_filt * 2, n_filt * 4)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8)
        self.indentity4 = downsample(n_filt * 4, n_filt * 8)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 16)
        self.deconv_1 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)
        self.Att5 = GridAttention2D(in_channels=1024, inter_channels=512, gating_channels=512, mode='concatenation',
                                    sub_factor=2)

        # self.deconv_1 = nn.Upsample(scale_factor=2)  # does only upscale width and height; Similar results to deconv2d

        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)
        self.Att4 = GridAttention2D(in_channels=512, inter_channels=256, gating_channels=256, mode='concatenation',
                                    sub_factor=2)
        # self.deconv_2 = nn.Upsample(scale_factor=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4, stride=1)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)
        self.Att3 = GridAttention2D(in_channels=256, inter_channels=128, gating_channels=128, mode='concatenation',
                                    sub_factor=2)
        # self.deconv_3 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_2 = nn.Conv2d(n_filt * 4 + n_filt * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # 'nearest' a bit faster but results a little worse (~0.4 dice points worse)
        self.output_2_up = nn.Upsample(scale_factor=2, mode=upsample)

        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2, stride=1)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)
        # self.deconv_4 = nn.Upsample(scale_factor=2)
        self.Att2 = GridAttention2D(in_channels=128, inter_channels=64, gating_channels=64, mode='concatenation',
                                    sub_factor=2)
        # Deep Supervision
        self.output_3 = nn.Conv2d(n_filt * 2 + n_filt * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_3_up = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height

        self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1)

        # no activation function, because is in LossFunction (...WithLogits)
        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inpt):
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_1_c = self.indentity1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        contr_1_2 = contr_1_2+contr_1_1_c
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_1_c = self.indentity2(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        contr_2_2 = contr_2_2+ contr_2_1_c
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_1_c = self.indentity3(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        contr_3_2 = contr_3_2+contr_3_1_c
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1(pool_3)
        contr_4_1_c = self.indentity4(pool_3)
        contr_4_2 = self.contr_4_2(contr_4_1)
        contr_4_2 = contr_4_2+contr_4_1_c
        pool_4 = self.pool_4(contr_4_2)

        if self.use_dropout:
            pool_4 = self.dropout(pool_4)

        encode_1 = self.encode_1(pool_4)
        encode_2 = self.encode_2(encode_1)
        deconv_1 = self.deconv_1(encode_2)
        deconv_1,Att_1 = self.Att5(deconv_1,contr_4_2)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1(concat1)
        expand_1_2 = self.expand_1_2(expand_1_1)
        deconv_2 = self.deconv_2(expand_1_2)
        deconv_2,Att_2 = self.Att4(deconv_2,contr_3_2)

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        deconv_3 = self.deconv_3(expand_2_2)
        deconv_3,Att_3 = self.Att3(deconv_3,contr_2_2)

        # Deep Supervision
        output_2 = self.output_2(concat2)
        output_2_up = self.output_2_up(output_2)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1(concat3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        deconv_4 = self.deconv_4(expand_3_2)
        deconv_4,Att_4 = self.Att2(deconv_4,contr_1_2)
        # Deep Supervision
        output_3 = output_2_up + self.output_3(concat3)
        output_3_up = self.output_3_up(output_3)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(concat4)
        expand_4_2 = self.expand_4_2(expand_4_1)

        conv_5 = self.conv_5(expand_4_2)

        final = output_3_up + conv_5

        return final
if __name__ == '__main__':
    model = AttUnet(n_input_channels=3,n_classes=1)
    inpt = torch.rand((1,3,384,384))
    out = model(inpt)
    print(out.size())







