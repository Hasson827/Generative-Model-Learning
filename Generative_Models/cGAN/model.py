import torch
from torch import nn
import torch.nn.functional as F

class Distinguish_Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # 标签嵌入层
        self.label_emb = nn.Embedding(num_classes, 50)
        
        # 图像处理分支
        self.image_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # 32*14*14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 64*7*7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.label_conv = nn.Sequential(
            nn.Linear(50, 7*7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.combined_conv = nn.Sequential(
            nn.Conv2d(64+1, 128, kernel_size=4, stride=2, padding=1), # 128*3*3
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0), # 256*1*1
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(256, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        x = x.view(-1, 1, 28, 28)  # 确保输入是28x28的图像
        
        # 处理图像
        img_features = self.image_conv(x) # [batch, 64, 7, 7]
        
        # 处理标签
        label_emb = self.label_emb(labels)  # [batch, 50]
        label_features = self.label_conv(label_emb) # [batch, 7*7]
        label_features = label_features.view(batch_size, 1, 7, 7)
        
        # 合并图像和标签特征
        combined = torch.cat((img_features, label_features), dim=1)
        
        # 最终判别
        combined = self.combined_conv(combined)
        return combined

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
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 标签嵌入层
        self.label_emb = nn.Embedding(num_classes, 50)
        
        # 生成网络
        self.gen = nn.Sequential(
            nn.Linear(latent_dim + 50, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.ReLU(inplace=True),
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
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        # 标签嵌入
        label_emb = self.label_embedding(labels)  # [batch, 50]
        
        # 拼接噪声和标签
        combined_input = torch.cat([noise, label_emb], dim=1)  # [batch, latent_dim + 50]
        
        # 生成过程
        x = self.fc(combined_input)
        x = x.view(-1, 256, 7, 7)
        x = self.res_block(x)
        x = self.upsample(x)
        return x

