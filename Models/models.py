import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from Models.layers import *
#from layers import *

     
class Lstm(torch.nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(8, 128, num_layers=3, dropout=0.5, bidirectional=True)        
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 100)
    def forward(self, x): 
        # input: B, C, L
        x = x.permute(2, 0, 1).contiguous() # L, B, C = (600, 100, 8)
        x, _ = self.lstm(x) # (600, 100, 256)
        
        x = x.permute(1, 2, 0).contiguous() # B, C, L = (100, 256, 600)
        mean = x.mean(axis=2) # (100, 600)
        std = x.std(axis=2) # (100, 600)
        x_cat = torch.cat((mean, std), 1) # (100, 1200)
        output = self.linear1(x_cat) # (100, 512)
        output = self.linear2(output) # (100, 100)
        return output, output

class Dense_Model(nn.Module):
    def __init__(self):
        super(Dense_Model, self).__init__()

        self.dense1 = Linear(7200, 784)
        self.dense2 = Linear(784, 1200)
        self.dense3 = Linear(1200, 1200)
        self.dense4 = Linear(1200, 10)
        self.register_forward_hook(forward_hook)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

class BiGRU_Model(nn.Module):
    def __init__(self):
        super(BiGRU_Model, self).__init__()

        self.bi_gru1 = nn.GRU(4,16,num_layers=2,dropout=0.3, batch_first=True,bidirectional=True)
        self.bi_gru2 = nn.GRU(32, 4, num_layers=1, dropout=0.3, batch_first=True, bidirectional=False)
        self.dense1 = Linear(4, 16)
        self.dense2 = Linear(16, 10)


    def forward(self, x):
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3],x.shape[-1]))
        x = x.permute(0,2,1)
       
        x, _ = self.bi_gru1(x)
        x, _ = self.bi_gru2(x)
        x = x[:,-1]
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class EnConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob=0.3, resi=False, time_reduce = True,dense=False):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.resi = resi
        self.dense = dense

        if time_reduce == True:
            self.conv_1 = Conv3d(self.in_chans, self.out_chans, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1,1,2))
        else:
            self.conv_1 = Conv3d(self.in_chans, self.out_chans, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1,1,1))
        self.BN = InstanceNorm3d(self.out_chans)
        self.actv = ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_prob)

        self.add = Add()

    def forward(self, inputs):

        
        out = self.conv_1(inputs)
        out = self.BN(out)
        out = self.actv(out)
        out = self.drop(out)

        if self.resi:
            out = self.add([inputs, out])
        if self.dense==False:
            return out
        else:
            return out, inputs

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
               f'drop_prob={self.drop_prob})'

    def relprop(self, R, alpha):
        R_1 = 0
        if self.resi:
            R_1, R = self.add.relprop(R, alpha)
        R = self.actv.relprop(R, alpha)
        R = self.BN.relprop(R, alpha)
        R = self.conv_1.relprop(R, alpha)
        R = R_1 + R
        return R


class UnetModel(nn.Module):
    def __init__(self, in_chans=1, chans=32, drop_prob=0.3):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            drop_prob (float): Dropout probability.
        """

        super().__init__(class_num=100)
        self.in_chans = in_chans
        self.chans = chans
        self.drop_prob = drop_prob
        self.ext_feature = EnConvBlock(self.in_chans, self.chans, drop_prob, resi=False)
        ch = self.chans
        self.en_level_1 = Sequential(EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True),
                                     EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True))
        self.down_1 = Sequential(Conv3d(ch, ch * 2, kernel_size=(3, 1, 3), padding=(1, 0, 1), stride=(2, 1, 2)),
                                 InstanceNorm3d(ch * 2), ReLU(inplace=True))  # (n,32,15,3,4)
        ch *= 2
        self.en_level_2 =Sequential(EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True),
                                     EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True))
        self.down_2 = Sequential(Conv3d(ch, ch * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 1, 2)),
                                 InstanceNorm3d(ch * 2), ReLU(inplace=False))  # (n,64,8,3,2)
        ch *= 2
        self.en_level_5 = Sequential(EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True),
                                     EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True))# (n,128,8,3,2)

        
        self.fc = Sequential(
            Linear(512, 256),
            # Linear(256, class_num),
            Linear(256, class_num)
        )
     
        
        


    def forward(self, x, rcam = False):

        ext_feature = self.ext_feature(x)  
        en_level_1 = self.en_level_1(ext_feature)
        down_1 = self.down_1(en_level_1)
        xn_level_2 = self.en_level_2(down_1) 
        down_2 = self.down_2(xn_level_2)
        en_level_5 = self.en_level_5(down_2)

        x_out = en_level_5.view(en_level_5.size(0),-1)

 
        x = self.fc(x_out)
        if rcam == False:
            return x, x_out

        R = self.CLRP(x)
        
        R = self.fc.relprop(R, alpha=1)
        R = R.reshape((R.size(0), 128, 1, 2, 2))
        R = self.en_level_5.relprop(R, alpha=1)
        R = self.down_2.relprop(R, alpha=1)
        R = self.en_level_2.relprop(R, alpha=1)
        R = self.down_1.relprop(R, alpha=1)
  
        r_weight = torch.mean(R, dim=(2, 3, 4), keepdim=True)
        r_cam = en_level_1 * r_weight
        r_cam = torch.sum(R,dim=1)

        r_cam = torch.reshape(r_cam,(r_cam.shape[0],8,-1))
        
        r_cam = r_cam.unsqueeze(0)
        r_cam = F.interpolate(r_cam, scale_factor=(1,600/r_cam.shape[-1]), mode='bicubic', align_corners=False)
        r_cam = r_cam.squeeze(0)

        return x, r_cam

    def CLRP(self,x):
        
        maxindex = torch.argmax(x,dim=-1)
        R = torch.ones(x.shape).cuda()
        
        R /= -1000
        R[:, maxindex] = 1
       

        return R

    def relprop(self, R):

        R = self.fc.relprop(R,alpha=1)
        R = R.reshape((R.size(0), 128,1,2,2))
        R = self.en_level_5.relprop(R,alpha=1)
        R = self.down_2.relprop(R,alpha=1)
        R = self.en_level_2.relprop(R,alpha=1)
        R = self.down_1.relprop(R,alpha=1)
        R = self.en_level_1.relprop(R,alpha=1)
        R = self.ext_feature.relprop(R,alpha=1)

        return R

class UnetModel2(nn.Module):
    def __init__(self, in_chans=1, chans=32, drop_prob=0.3,class_num=100):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            drop_prob (float): Dropout probability.
        """

        super().__init__()
        self.in_chans = in_chans
        self.chans = chans
        self.drop_prob = drop_prob
        self.ext_feature = EnConvBlock(self.in_chans, self.chans, drop_prob, resi=False)
        ch = self.chans
        self.en_level_1 = Sequential(EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True),
                                     EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True))
        self.down_1 = Sequential(Conv3d(ch, ch * 2, kernel_size=(3, 1, 3), padding=(1, 0, 1), stride=(2, 1, 2)),
                                 InstanceNorm3d(ch * 2), ReLU(inplace=True))  # (n,32,15,3,4)
        ch *= 2
        self.en_level_2 =Sequential(EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True),
                                     EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True))
        self.down_2 = Sequential(Conv3d(ch, ch * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 1, 2)),
                                 InstanceNorm3d(ch * 2), ReLU(inplace=False))  # (n,64,8,3,2)
        ch *= 2
        self.en_level_5 = Sequential(EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True),
                                     EnConvBlock(ch, ch, drop_prob,resi=False,time_reduce=True))# (n,128,8,3,2)

        self.down_3 = Sequential(Conv3d(ch, ch, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 2)),
                                 InstanceNorm3d(ch), ReLU(inplace=False),
                                 Conv3d(ch, ch, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 2)))

        
        self.fc = Sequential(
            Linear(512, 256),
            # Linear(256, 100),
            Linear(256, class_num)
        )
     
        

    def forward(self, x, rcam = False):
        
        ext_feature = self.ext_feature(x)
        en_level_1 = self.en_level_1(ext_feature)
        down_1 = self.down_1(en_level_1)
        xn_level_2 = self.en_level_2(down_1)
        down_2 = self.down_2(xn_level_2)
        en_level_5 = self.en_level_5(down_2)
        down_3 = self.down_3(en_level_5)

        x_out = down_3.view(down_3.size(0),-1)
        x = self.fc(x_out)
        if rcam == False:
            return x, x_out

        R = self.CLRP(x)
        
        R = self.fc.relprop(R, alpha=1)
        R = R.reshape((R.size(0), 128, 1, 2, 2))
        R = self.en_level_5.relprop(R, alpha=1)
        R = self.down_2.relprop(R, alpha=1)
        R = self.en_level_2.relprop(R, alpha=1)
        R = self.down_1.relprop(R, alpha=1)
  
        r_weight = torch.mean(R, dim=(2, 3, 4), keepdim=True)
        r_cam = en_level_1 * r_weight
        r_cam = torch.sum(R,dim=1)

        r_cam = torch.reshape(r_cam,(r_cam.shape[0],8,-1))
        
        r_cam = r_cam.unsqueeze(0)
        r_cam = F.interpolate(r_cam, scale_factor=(1,600/r_cam.shape[-1]), mode='bicubic', align_corners=False)
        r_cam = r_cam.squeeze(0)

        return x, r_cam

    def CLRP(self,x):
        
        maxindex = torch.argmax(x,dim=-1)
        R = torch.ones(x.shape).cuda()
        
        R /= -1000
        R[:, maxindex] = 1
       

        return R

    def relprop(self, R):

        R = self.fc.relprop(R,alpha=1)
        R = R.reshape((R.size(0), 128,1,2,2))
        R = self.en_level_5.relprop(R,alpha=1)
        R = self.down_2.relprop(R,alpha=1)
        R = self.en_level_2.relprop(R,alpha=1)
        R = self.down_1.relprop(R,alpha=1)
        R = self.en_level_1.relprop(R,alpha=1)
        R = self.ext_feature.relprop(R,alpha=1)

        return R

if __name__ == "__main__":
    import os
    from torchinfo import summary
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetModel().to(device)

    summary(model, (8,1,4,1,2000))








