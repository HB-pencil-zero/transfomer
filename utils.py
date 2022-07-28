from tracemalloc import start
import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as func 
    
def attn_mecanism(K:Tensor,Q:Tensor,V:Tensor):
    attn=Q@K.t
    attn/=torch.sqrt( attn.shape[-1]).float()
    attn=torch.softmax(attn,-1)
    return attn@V

#单层的attentionblock机制
class attnBlock(nn.Module):
    def __init__(self,dim_input:int,dim_val:int,dim_attn:int,c:int=1):
        super(attnBlock,self).__init__()
        self.modulelist=nn.ModuleList()
        for i in range(2):
            self.modulelist.append(nn.Linear(dim_input,dim_attn))
        self.modulelist.append(nn.Linear(dim_input ,dim_val))
        self.bn=nn.BatchNorm2d(c)

    def forward(self,x:Tensor,kv=None):
        if(kv is not None):
            k,q=self.modulelist[0](x),self.modulelist[1](x)
            v=self.modulelist[2](x)
            return self.bn(attn_mecanism(k,q,v))
        k,q,v=self.modulelist[0](x),self.modulelist[1](x),self.modulelist[2](x)
        mx=self.bn(mx)
        return mx
    

        



class multiAttnBlock(nn.Module):
    def __init__(self,inFeature:int,outFeature:int,dim_val:int,dim_attn:int,numHeaders:int,c:int=1):
        super(multiAttnBlock,self).__init__()
        self.inFeature,self.outFeature,self.dim_val,self.dim_attn,self.numHeaders=list(inFeature,outFeature,dim_val,dim_attn,numHeaders)
        self.moudlelist=nn.ParameterList()
        for i in range(numHeaders):
            self.moudlelist.append(attnBlock(inFeature,dim_val,dim_attn,c))
        self.linear=nn.Linear(numHeaders*dim_val,outFeature)

    def forward(self,X:Tensor,kv=None):
        vector_list=[]
        for module in self.moudlelist:
            vector_list.append(module(X,kv))
        mx=torch.stack(vector_list,dim=-1).flatten(start_dim=-2)
        mx=self.linear(mx)
        return mx
        