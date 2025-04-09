import torch
from torch import nn

class Distinguish_Model(nn.Module):
    """
    This model is used to distinguish between real and fake data.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = self.model(x)
        return x

class Generate_Model(nn.Module):
    """
    This model is used to generate fake data.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self,x):
        x = self.model(x)
        return x

