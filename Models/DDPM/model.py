import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class ResidualBlock(nn.Module):
    """
    A Residual Block with two convolutional layers and a skip connection.
    Input: (batch_size, in_channels, height, width)
    Output: (batch_size, out_channels, height, width)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # 残差路径（如果输入输出通道不同，需要投影）
        self.residual_path = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        h = x
        h = F.relu(self.norm1(self.conv1(h)))
        h = self.norm2(self.conv2(h))
        return F.relu(h + self.residual_path(x))

class DownBlock(nn.Module):
    """
    A Downsampling Block with two Residual Blocks and a downsampling layer.
    Input: (batch_size, in_channels, height, width)
    Output: (batch_size, out_channels, height/2, width/2)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, in_channels)
        self.res2 = ResidualBlock(in_channels, in_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return self.downsample(x), x  # 返回下采样结果和跳跃连接

class UpBlock(nn.Module):
    """
    An Upsampling Block with two Residual Blocks and an upsampling layer.
    Input: (batch_size, in_channels, height/2, width/2)
    Output: (batch_size, out_channels, height, width)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels + in_channels//2, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        return self.res2(x)

class SelfAttention(nn.Module):
    """
    A Self-Attention Block with Multihead Attention and Feedforward layers.
    Input: (batch_size, channels, height, width)
    Output: (batch_size, channels, height, width)
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, H*W, C
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(1, 2).reshape(-1, self.channels, *size)

class TimeEmbedding(nn.Module):
    """
    A Time Embedding Block that uses sinusoidal position embeddings.
    Input: (batch_size, 1)
    Output: (batch_size, dim)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 使用正弦位置编码，而不是简单的线性层
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim // 4),
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, t):
        return self.mlp(t)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal Position Embeddings for time steps.
    Input: (batch_size, 1)
    Output: (batch_size, dim)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings

class UNet(nn.Module):
    def __init__(self, input_channels=1, hidden_dims=128):
        super().__init__()
        
        # 初始卷积
        self.init_conv = nn.Conv2d(input_channels, hidden_dims, kernel_size=3, padding=1)
        
        # 下采样路径
        self.down1 = DownBlock(hidden_dims, hidden_dims*2)
        self.down2 = DownBlock(hidden_dims*2, hidden_dims*4)
        self.down3 = DownBlock(hidden_dims*4, hidden_dims*8)
        
        # 中间层 (带自注意力)
        self.mid_attn = SelfAttention(hidden_dims*8)
        self.mid_block1 = ResidualBlock(hidden_dims*8, hidden_dims*8)
        self.mid_block2 = ResidualBlock(hidden_dims*8, hidden_dims*8)
        
        # 标签条件嵌入
        self.label_embedding = nn.Embedding(10, hidden_dims*8)
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(hidden_dims*8)
        
        # 上采样路径 (带跳跃连接)
        self.up1 = UpBlock(hidden_dims*8, hidden_dims*4)
        self.up2 = UpBlock(hidden_dims*4, hidden_dims*2)
        self.up3 = UpBlock(hidden_dims*2, hidden_dims)
        
        # 注意力层 (在多个分辨率层级)
        self.attn1 = SelfAttention(hidden_dims*4)
        self.attn2 = SelfAttention(hidden_dims*2)
        
        # 输出层
        self.final_res = ResidualBlock(hidden_dims, hidden_dims)
        self.final_conv = nn.Conv2d(hidden_dims, input_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, labels=None):
        """
        Input:
            x (torch.Tensor): Input image tensor. shape: (batch_size, channels, height, width)
            t (torch.Tensor): Time step tensor. shape: (batch_size,)
            labels (torch.Tensor, optional): Labels for conditional generation. shape: (batch_size,10)
        """
        # 初始特征提取
        x = self.init_conv(x) # (batch_size, hidden_dims, height, width)
        
        # 下采样并存储跳跃连接
        x1, skip1 = self.down1(x) # (batch_size, hidden_dims*2, height/2, width/2)
        x2, skip2 = self.down2(x1) # (batch_size, hidden_dims*4, height/4, width/4)
        x3, skip3 = self.down3(x2) # (batch_size, hidden_dims*8, height/8, width/8)
        
        # 时间嵌入
        t_emb = self.time_embedding(t.float().unsqueeze(-1)) # (batch_size, hidden_dims*8)
        
        # 标签条件嵌入 (如果有)
        if labels is not None:
            label_emb = self.label_embedding(labels) # (batch_size, hidden_dims*8)
            context = t_emb + label_emb # (batch_size, hidden_dims*4)
        else:
            context = t_emb
        # 中间处理
        h = self.mid_block1(x3) # (batch_size, hidden_dims*8, height/8, width/8)
        h = self.mid_attn(h) + h  # 残差自注意力
        h = self.mid_block2(h) # (batch_size, hidden_dims*8, height/8, width/8)
        _,_,h_dim,w_dim = h.shape
        # 添加条件信息
        context = context.repeat(1,1,h_dim,w_dim) # (batch_size, hidden_dims*8, height/8, width/8)
        
        # 上采样并使用跳跃连接
        h = self.up1(h, skip3) # (batch_size, hidden_dims*4, height/4, width/4)
        h = self.attn1(h) + h  # 残差自注意力
        h = self.up2(h, skip2) # (batch_size, hidden_dims*2, height/2, width/2)
        h = self.attn2(h) + h  # 残差自注意力
        h = self.up3(h, skip1) # (batch_size, hidden_dims, height, width)
        
        # 输出处理
        h = self.final_res(h)
        return self.final_conv(h)


class DDPM(nn.Module):
    """
    A Denoising Diffusion Probabilistic Model (DDPM) for image generation.
    """
    def __init__(self,n_steps=1000,beta_start=1e-4,beta_end=0.02):
        """
        Args:
            n_steps (int): Number of diffusion steps.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
        """
        super().__init__()
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Noise Schedule
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-Calculate Values for Inference
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def to(self,device):
        """
        Move model parameters to the specified device.
        
        Args:
            device (torch.device): Device to move the model to.
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self
    
    def add_noise(self,x_0,t):
        """
        Add noise to the input image.
        
        Args:
            x_0 (torch.Tensor): Input image.
            t (torch.Tensor): Time step.
        
        Returns:
            torch.Tensor: Noisy image.
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        noisy_image = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return noisy_image, noise
    
    def sample(self,model,n_samples,img_size,device, labels=None):
        """
        Sample images from the model.
        
        Args:
            model (nn.Module): The UNet model.
            n_samples (int): Number of samples to generate.
            img_size (int): Size of the generated images.
            device (torch.device): Device to run the model on.
            labels (torch.Tensor, optional): Labels for conditional generation.
        
        Returns:
            torch.Tensor: Generated images.
        """
        x = torch.randn(n_samples,1,img_size,img_size).to(device)
            
        # Denoising Loop
        for t in tqdm(reversed(range(self.n_steps)),desc="Sampling"):
            t_batch = torch.ones(n_samples,dtype=torch.long).to(device) * t
            predicted_noise = model(x,t_batch,labels)
                
            alpha = self.alphas[t].to(device)
            alpha_cumprod = self.alphas_cumprod[t].to(device)
            beta = self.betas[t].to(device)
                
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            tilde_mu = (1/torch.sqrt(alpha)) * (x - (beta/torch.sqrt(1-alpha_cumprod)*predicted_noise))
            x = tilde_mu + torch.sqrt(beta) * noise
        
        return x