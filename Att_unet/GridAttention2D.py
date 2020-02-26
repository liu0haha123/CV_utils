import torch
import torch.nn as nn
import torch.nn.functional as F


class GridAttention2D(nn.Module):
    def __init__(self,in_channels,gating_channels,inter_channels,mode="concate",sub_factor=2):
        super(GridAttention2D, self).__init__()
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        self.mode = mode

        if self.inter_channels==None:
            self.inter_channels = in_channels//2
            if self.inter_channels==0:
                self.inter_channels=1

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_channels)
        )

        self.theta = nn.Conv2d(in_channels=self.in_channels,out_channels=self.inter_channels,
                               kernel_size=sub_factor,stride=sub_factor,padding=0,bias=False)

        self.phi = nn.Conv2d(in_channels=self.gating_channels,out_channels=self.inter_channels,
                             kernel_size=1,stride=1,padding=0,bias=False)

        self.psi = nn.Conv2d(in_channels=self.inter_channels,out_channels=1,stride=1,kernel_size=1,padding=0,bias=True)


    def forward(self,X,g):

        input_size = X.size()
        batch_size = input_size[0]

        assert batch_size==g.size(0)

        theta_x =self.theta(X)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g),size=theta_x_size[2:],mode="bilinear",align_corners=True)
        f= F.relu(phi_g+theta_x,inplace=True)
        sigmoid_psi_f = torch.sigmoid(self.psi(f))
        sigmoid_psi_f = F.interpolate(sigmoid_psi_f,size=input_size[2:],mode="bilinear",align_corners=True)
        y = sigmoid_psi_f.expand_as(X)*X
        W_y = self.W(y)

        return W_y,sigmoid_psi_f

if __name__ == '__main__':
    from torch.autograd import Variable

    mode_list = ['concatenation']

    for mode in mode_list:

        img = Variable(torch.rand(3, 512,  36, 36))
        gat = Variable(torch.rand(3, 256, 18, 18))
        net = GridAttention2D(in_channels=512, inter_channels=512, gating_channels=256, mode='concatenation', sub_factor=2)
        out, sigma = net(img, gat)
        print(out.size(),sigma.size())
