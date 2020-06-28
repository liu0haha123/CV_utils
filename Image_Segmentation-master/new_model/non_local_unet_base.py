import torch
import torch.nn as nn
import torch.nn.functional as F


class input_Block(nn.Module):
    def __init__(self,input_channel,output_channel):
        # mode 控制模型的阶段，训练阶段需要BN,推断阶段没有
        super(input_Block, self).__init__()
        self.bnRelu6 = nn.Sequential(
            nn.BatchNorm2d(input_channel,momentum=0.997,eps=1e-5),
            nn.ReLU6(inplace=True)
            )
        self.conv2d = nn.Conv2d(in_channels = input_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        self.identity = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,stride=1,padding=0)

    def forward(self,input):
        res = self.identity(input)
        out = self.conv2d(self.bnRelu6(input))
        out = self.conv2d(self.bnRelu6(input))
        out = out+res
        return out


class downsample_Resdiual_Block(nn.Module):
    def __init__(self,input_channel):
        super(downsample_Resdiual_Block, self).__init__()
        self.bnRelu6_1 = nn.Sequential(nn.BatchNorm2d(input_channel),nn.ReLU6(inplace=True))
        self.bnRelu6_2 = nn.Sequential(nn.BatchNorm2d(2*input_channel), nn.ReLU6(inplace=True))
        self.conv2dC3S1 =nn.Conv2d(in_channels = 2*input_channel,out_channels=2*input_channel,kernel_size=3,stride=1,padding=1)
        self.conv2dC3S2 = nn.Conv2d(in_channels = input_channel,out_channels=2*input_channel,kernel_size=3,stride=2,padding=1)
        self.conv2dC1S2 = nn.Conv2d(in_channels = input_channel,out_channels=2*input_channel,kernel_size=1,stride=2,padding=0)


    def forward(self,input):
       res = self.conv2dC1S2(input)
       out = self.conv2dC3S2(self.bnRelu6_1(input))
       out = self.conv2dC3S1(self.bnRelu6_2(out))

       out = out+res

       return  out


class Multi_Head_Attention(nn.Module):
    def __init__(self, in_channel, key_filters, value_filters, output_filters, num_heads, dropout=0.5,
                 query_transform_type='SAME'):
        """

        :param in_channel: 输入张量的通道数[batch_size,H,W,channel]
        :param key_filters: key部分的filter数
        :param value_filters:
        :param output_filters:
        :param num_heads:
        :param dropout:
        :param query_transform_type:
        """
        super(Multi_Head_Attention, self).__init__()
        if key_filters % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (key_filters, num_heads))
        if value_filters % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (value_filters, num_heads))
        if query_transform_type not in ['SAME', 'DOWN', 'UP', 'Arbitrary']:
            raise ValueError("Layer type (%s) must be one of SAME, "
                             "DOWN, UP, Arbitrary." % (query_transform_type))
        self.num_head = num_heads
        self.att_dropout = nn.Dropout(dropout)
        self.query_transform_type = query_transform_type
        if query_transform_type == "SAME":
            # 等距计算self_attention
            self.QueryTransform = nn.Conv2d(in_channels=in_channel,out_channels=key_filters,kernel_size=1, stride=1,
                                padding=0, bias=True)
        elif query_transform_type == "DOWN":
            #降采样
            self.QueryTransform = nn.Conv2d(in_channel, key_filters, kernel_size=3, stride=2,
                                padding=1, bias=True)
        elif query_transform_type == "UP":
            self.QueryTransform = nn.ConvTranspose2d(in_channel, key_filters, kernel_size=3, stride=2,
                                padding=1, bias=True)
        elif query_transform_type == 'Arbitrary':
             self.QueryTransform = nn.Conv2d(in_channel, key_filters, kernel_size=1, stride=1,padding=0, bias=True)

        self.keyQueryTransform = nn.Conv2d(in_channel,key_filters,kernel_size=1, stride=1, padding=0, bias=True)
        self.valueQueryTransform = nn.Conv2d(in_channel, value_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.outputTransform = nn.Conv2d(value_filters,output_filters,kernel_size=1, stride=1, padding=0, bias=True)
        # 计算 self-attention时的scale
        self._scale = (key_filters // num_heads) ** 0.5

    def forward(self,input,Arbitrary_shape = None):
        # Arbitrary_shape:输出张量的维度[h,w]
        # input:[batch_size,C,H,W]
        if self.query_transform_type =="SAME" or self.query_transform_type=="DOWN":
            q = self.QueryTransform(input)
        elif self.query_transform_type =="UP":
            q = self.QueryTransform(input, output_size=(input.shape[2]*2, input.shape[3]*2))

        elif self.query_transform_type =="Arbitrary":
            q = self.QueryTransform(input)
            if Arbitrary_shape == None:
                Arbitrary_shape = (q.shape[2], q.shape[3])
            q = F.interpolate(q, size=Arbitrary_shape, mode='bilinear', align_corners=False)

        k = self.keyQueryTransform(input).permute(0,2,3,1)
        v = self.valueQueryTransform(input).permute(0,2,3,1)
        q = q.permute(0, 2, 3, 1)
        #q:[batch_size,H,W,C]
        batch_size = q.shape[0]
        Hq =q.shape[1]
        Wq = q.shape[2]
        k = self.split_heads(k, self.num_head)
        v = self.split_heads(v, self.num_head)
        q = self.split_heads(q, self.num_head)
        # 论文中的unfold
        #[(B, H, W, N), c]
        k = torch.flatten(k, 0, 3)
        v = torch.flatten(v, 0, 3)
        q = torch.flatten(q, 0, 3)

        # scale
        q = q/self._scale
        A = torch.matmul(q, k.transpose(0, 1))
        A = torch.softmax(A, dim=1)
        A = self.att_dropout(A)

        # [(B, Hq, Wq, N), C]
        O = torch.matmul(A, v)
        # fold
        O = O.view(batch_size, Hq, Wq, v.shape[-1]*self.num_head)
        O = O.permute(0, 3, 1, 2)
        O = self.outputTransform(O)

        return O


    def split_heads(self,input,num_head):
        """
        将 channel分为多个head进行多头注意力的计算
        :param input: #q:[batch_size,H,W,C]
        :param num_head:
        :return:out：[batch_size,H,W,num_head,C//num_head]
        """
        C = input.shape[-1]
        out = input.view(input.shape[0], input.shape[1], input.shape[2], num_head, int(C/num_head))

        return out

class bottomBlock(nn.Module):
    def __init__(self,in_channel, key_filters, value_filters, output_filters, num_heads, dropout=0.5,
                 query_transform_type='SAME'):
        super(bottomBlock, self).__init__()
        self.globalAggreationBlock = Multi_Head_Attention(in_channel, key_filters, value_filters, output_filters, num_heads, dropout=dropout,
                 query_transform_type=query_transform_type)
    def forward(self,input):
        out = self.globalAggreationBlock(input)
        out = out+input
        return out


"""
class upsamplingBlock(nn.Module):
    def __init__(self,in_channel,key_filters, value_filters, output_filters, num_heads, dropout,
                 query_transform_type='UP'):
        super(upsamplingBlock, self).__init__()
        self.globalAggreationBlock = Multi_Head_Attention(in_channel, key_filters, value_filters, output_filters, num_heads, dropout=dropout,
                 query_transform_type=query_transform_type)
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel,out_channels=output_filters,kernel_size=3,stride=2,output_padding=1,padding=1)



    def forward(self,input):
        res = self.deconv(input)
        out = self.globalAggreationBlock(input)
        out = res+out
        return out

"""
class upsamplingBlock(nn.Module):
    def __init__(self,in_channel,key_filters,value_filters,output_filters,num_heads,dropout,query="UP"):
        super(upsamplingBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel,out_channels=output_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.globalAggreationBlock = Multi_Head_Attention(in_channel=in_channel,key_filters=key_filters,value_filters=value_filters,output_filters=output_filters,
                                                          num_heads=num_heads,dropout=dropout,query_transform_type=query)

    def forward(self,inputs):
        res = self.deconv(inputs)
        output = self.globalAggreationBlock(inputs)
        output = res+output
        return output



class output_Block(nn.Module):
    def __init__(self,in_channel,num_classes,dropout):
        super(output_Block, self).__init__()
        self.bnRelu6 = nn.Sequential(
            nn.BatchNorm2d(in_channel, momentum=0.997, eps=1e-5),
            nn.ReLU6(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=num_classes,kernel_size=1,stride=1,bias=True)

    def forward(self,inputs):
        out = self.bnRelu6(inputs)
        out = self.dropout(out)
        out = self.conv(out)

        return out

class non_local_Unet(nn.Module):
    def __init__(self,num_classes,in_channel,key_filters,value_filters, num_heads, dropout=0.35):
        """

        :param num_classes:
        :param in_channel:
        :param key_filters:
        :param value_filters:
        :param output_filters:
        :param num_heads:
        :param dropout:
        """
        super(non_local_Unet, self).__init__()
        self.input_Block = input_Block(input_channel=in_channel,output_channel=32)
        self.downSamplingBlock1 = downsample_Resdiual_Block(input_channel=32)
        self.downSamplingBlock2 = downsample_Resdiual_Block(input_channel=64)
        self.bottomBlock = bottomBlock(in_channel=128,key_filters=8,output_filters=128,value_filters=8,num_heads=num_heads,dropout=dropout)
        self.upSamplingBlock2 = upsamplingBlock(in_channel=128,key_filters=key_filters,output_filters=64,value_filters=value_filters,num_heads=num_heads,dropout=dropout)
        self.upSamplingBlock1 = upsamplingBlock(in_channel=64,key_filters=key_filters,output_filters=32,value_filters=value_filters,num_heads=num_heads,dropout=dropout)
        self.outputBlock = output_Block(in_channel=32,num_classes=num_classes,dropout=dropout)
    def forward(self,inputs):
        inputs = self.input_Block(inputs)
        D1 = self.downSamplingBlock1(inputs)
        D2 = self.downSamplingBlock2(D1)
        bottom = self.bottomBlock(D2)
        U2 = self.upSamplingBlock2(bottom)
        U2 = U2+D1
        U1 = self.upSamplingBlock1(U2)
        U1 = U1+inputs
        output = self.outputBlock(U1)


        return output