import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class UNet(nn.Module):
    """
    A UNet for the denoising model.
    """
    def __init__(self,input_channels=1,hidden_dims=64):
        super().__init__()
        
        self.init_conv = nn.Conv2d(input_channels, hidden_dims, kernel_size=3, padding=1)
        
        # Downsampling
        self.down1 = nn.Conv2d(hidden_dims, hidden_dims*2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(hidden_dims*2, hidden_dims*4, kernel_size=4, stride=2, padding=1)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dims*4),
            nn.ReLU(),
            nn.Linear(hidden_dims*4, hidden_dims*4)
        )
        
        # Unsampling
        self.up1 = nn.ConvTranspose2d(hidden_dims*4, hidden_dims*2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(hidden_dims*2, hidden_dims, kernel_size=4, stride=2, padding=1)
        
        self.final_conv = nn.Conv2d(hidden_dims, input_channels, kernel_size=3, padding=1)
        
    def forward(self,x,t):
        x1 = F.relu(self.init_conv(x))
        x2 = F.relu(self.down1(x1))
        x3 = F.relu(self.down2(x2))
        
        t = t.float().unsqueeze(-1)
        t = self.time_mlp(t)
        t = t.view(-1,t.shape[1],1,1).expand(-1,-1,x3.shape[2],x3.shape[3])
        x3 = x3 + t
        
        x = F.relu(self.up1(x3))
        x = F.relu(self.up2(x))
        x = self.final_conv(x)
        
        return x


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
    
    def sample(self,model,n_samples,img_size,device):
        """
        Sample images from the model.
        
        Args:
            model (nn.Module): The UNet model.
            n_samples (int): Number of samples to generate.
            img_size (int): Size of the generated images.
            device (torch.device): Device to run the model on.
        
        Returns:
            torch.Tensor: Generated images.
        """
        model.eval()
        with torch.inference_mode():
            x = torch.randn(n_samples,1,img_size,img_size).to(device)
            
            # Denoising Loop
            for t in tqdm(reversed(range(self.n_steps)),desc="Sampling"):
                t_batch = torch.ones(n_samples,dtype=torch.long).to(device) * t
                predicted_noise = model(x,t_batch)
                
                alpha = self.alphas[t].to(device)
                alpha_cumprod = self.alphas_cumprod[t].to(device)
                beta = self.betas[t].to(device)
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                tilde_mu = (1/torch.sqrt(alpha)) * (x - (beta/torch.sqrt(1-alpha_cumprod)*predicted_noise))
                x = tilde_mu + torch.sqrt(beta) * noise
        
        model.train()
        return x

def test_ddpm():
    """
    Test the DDPM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.mps.is_available()
                        else "cpu")
    model = UNet().to(device)
    ddpm = DDPM(n_steps=1000).to(device)
    
    print("Tesing forward noise addition")
    x_0 = torch.randn(4,1,32,32).to(device)
    t = torch.randint(0,1000,(4,)).to(device)
    noisy_images,noise = ddpm.add_noise(x_0,t)
    assert noisy_images.shape == x_0.shape, f"Shape Mismatch: {noisy_images.shape} != {x_0.shape}"
    print("Forward noise addition test passed.")
    
    print("\n Tesing model forward pass")
    predicted_noise = model(noisy_images,t)
    assert predicted_noise.shape == noisy_images.shape, f"Shape Mismatch: {predicted_noise.shape} != {noisy_images.shape}"
    print("Model forward pass test passed.")
    
    print("\n Testing sampling process")
    samples = ddpm.sample(model,n_samples=2,img_size=32,device=device)
    assert samples.shape == (2,1,32,32), f"Shape Mismatch: {samples.shape} != {(2,1,32,32)}"
    assert not torch.isnan(samples).any(), "Generated samples contain NaN values."
    print("Sampling process test passed.")
    
    print("\n All tests passed.")
    
    
if __name__ == "__main__":
    test_ddpm()

"""
DDPM模型宏观流程解析
DDPM(去噪扩散概率模型)是一种强大的生成模型,其基本原理可以分为"前向扩散"和"反向去噪"两个主要过程。

核心思想
DDPM的核心思想非常直观:
    前向过程：逐步给图像添加噪声，直到完全破坏原始图像信息
    反向过程：学习如何从噪声中恢复图像，从而实现生成新图像

模型架构
    代码中包含两个关键组件：
        UNet模型:负责预测噪声的神经网络
            典型的U形编码器-解码器架构
            包含时间步嵌入，使网络知道当前去噪的阶段
        DDPM类:实现扩散过程的逻辑
            定义噪声调度(从beta_start到beta_end)
            实现前向加噪和反向采样的方法

前向扩散过程
在训练阶段,前向扩散过程通过add_noise方法实现:
    逐渐向原始图像x₀添加高斯噪声
    噪声程度由预定义的调度控制(beta值)
    数学表示:x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * ε，其中ε是随机噪声

反向去噪过程
在生成(推理)阶段,反向去噪通过sample方法实现:
    从纯高斯噪声x_T开始
    逐步反向遍历时间步(从T到0)
    在每个时间步：
        使用UNet预测当前噪声
        根据预测噪声计算更清晰的图像
        添加少量随机噪声（除了最后一步）以增加多样性
        最终得到生成图像x₀
        
训练目标
虽然代码中没有显式包含训练部分,但DDPM的训练目标是:
    给定噪声图像x_t和时间步t
    使UNet模型学习预测原始添加的噪声
    最小化预测噪声与实际添加噪声之间的误差
    
总结
DDPM通过将图像生成建模为逐步去噪的过程,实现了高质量的图像生成。
这种方法直观且有效，已成为现代生成模型的重要基础。
测试代码验证了模型的三个关键功能：前向加噪、模型预测和采样过程，确保了实现的正确性。
"""