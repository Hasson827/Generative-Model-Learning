import torch
from torch import nn

class Distinguish_Model(nn.Module):
    """
    This model is used to distinguish between real and fake data.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # 32*14*14
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 64*7*7
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128*4*4
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0), # 256*1*1
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Flatten(),
            nn.Linear(256, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = x.view(-1, 1, 28, 28)
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x_init = x
        out = self.block(x)
        out += x_init
        return self.relu(out)

class Generate_Model(nn.Module):
    """
    This model is used to generate fake data.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.ReLU(inplace=True)
        )
        
        self.res_block = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 128*14*14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64*28*28
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = self.res_block(x)
        x = self.upsample(x)
        return x