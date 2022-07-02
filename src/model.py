import torch
from torch import nn


''' One block of the net: 1D conv -> BatchNorm -> Activation '''
class Block(nn.Module):
    
    def __init__(self, chIn, chOut, kernel):
        super(Block, self).__init__()
        
        self.conv = nn.Conv1d(chIn, chOut, kernel,
                              padding = (kernel - 1) // 2,
                              padding_mode = 'reflect')
        self.bn   = nn.BatchNorm1d(chOut)
        self.act  = nn.SiLU()
        
        self.layers = nn.ModuleList([self.conv, self.bn, self.act])
    
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x

''' Encoder: Stacks multiple blocks '''
class Encoder(nn.Module):
    def __init__(self):
        
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            Block(chIn = 3,   chOut = 32,  kernel = 128),
            Block(chIn = 32,  chOut = 64, kernel = 64),
            nn.MaxPool1d(kernel_size = 2),
            Block(chIn = 64,  chOut = 128,  kernel = 32),
            Block(chIn = 128, chOut = 256,  kernel = 16),
            nn.MaxPool1d(kernel_size = 2)
        ])

    def forward(self, x):
        
        for layer in self.layers: 
            x = layer(x)
        
        return x
    
''' Final network: encoder -> Global avg. pooling -> linear layer '''
class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()
        
        self.encoder = Encoder()
        self.linear  = nn.Linear(256, 1)
    
    def forward(self, x):
        
        x = self.encoder(x)
        x = x.mean(dim = 2)
        x = self.linear(x)
        return x