import torch 
import torch.nn as nn 
from utils import *
from torch import Tensor

class Encoder(nn.Module):
    def __init__(self,inFeature:int,hidFeature:int,outFeature,dim_v:int,dim_attn:int,c:int=1):
        super(Encoder ,self).__init__()
        self.multiAttn=multiAttnBlock(inFeature,hidFeature,dim_v,dim_attn,c)
        self.Linear=nn.Linear(hidFeature,outFeature)
        self.bn1=nn.BatchNorm2d(c)
        self.bn2=nn.BatchNorm2d(c)

    def forward(self,X:Tensor):
        X=X+self.multiAttn(X)
        X=self.bn1(X)
        X=X+self.Linear(X)
        X=self.bn2(X)
        return X 

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()