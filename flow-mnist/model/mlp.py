import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    def __init__(self,input_dim:int, hidden_dims:List[int], output_dim:int, act=nn.ReLU()):
        super(MLP, self).__init__()
        # construct an MLP net
        nets= []
        nets.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims)-1:
                nets.append(act)
                nets.append(nn.Linear(hidden_dims[i],output_dim))
            else:
                nets.append(act)
                nets.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
        self.net=nn.Sequential(*nets)
    
    def forward(self,x):
        for layer in self.net:
            x=layer(x)
        return x
    
        
        