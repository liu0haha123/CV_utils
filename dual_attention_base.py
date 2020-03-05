import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import  Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d,\
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
import torch.nn.functional as F

from torch.autograd import Variable


class PAM_moudle(Module):
    """
    Position attention module
    """
    def __init__(self,in_dim):
        super(PAM_moudle, self).__init__()
        self.in_dim = in_dim
        self.query_conv = Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

        self.Softmax = Softmax(dim=-1)

    def forward(self,X):
        """

        :param X:(B* C* H * W)
        :return:out=attention+input feature
        attention:(B*(H*W)*(H*W))
        """
        batch_size,C,H,W = X.size()
        proj_query = self.query_conv(X).view(batch_size,-1,H*W).permute(0,2,1)
        proj_key = self.key_conv(X).view(batch_size,-1,H*W)

        attention = self.Softmax(torch.bmm(proj_query,proj_key))
        proj_value = self.value_conv(X).view(batch_size,-1,H*W)

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(batch_size,C,H,W)

        out = self.gamma*out+X

        return X

class CAM_moudle(Module):
    def __init__(self,in_dim):
        super(CAM_moudle, self).__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.Softmax = Softmax(dim=-1)

    def forward(self,X):
        batch_size,C,H,W = X.size()

        proj_query = X.view(batch_size,C,-1)
        proj_key = X.view(batch_size,C,-1).permute(0,2,1)
        energy = torch.bmm(proj_query,proj_key)
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.Softmax(energy_new)

        proj_value = X.view(batch_size,C,-1)
        out= torch.bmm(attention,proj_value)
        out = out.view(batch_size,C,H,W)

        out = self.gamma*out+X
        return out
if __name__ == '__main__':
    test_tensor = Variable(torch.rand((4,64,72,72)))
    PAM = CAM_moudle(in_dim=64)
    out = PAM(test_tensor)
    print(out.size())
