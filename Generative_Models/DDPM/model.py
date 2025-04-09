import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码，用于将时间步嵌入为向量表示
    这使得UNet能够知道当前处于哪个噪声水平
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    UNet的基本构建块,包含卷积、批归一化和激活函数
    特征提取:通过第一个卷积层、批归一化和ReLU激活函数处理输入
    时间条件融合：
        处理时间嵌入
        将时间向量的维度从 [batch_size, out_ch] 扩展为 [batch_size, out_ch, 1, 1]
        这个维度扩展使时间信息可以通过广播机制与所有空间位置的特征相加
        这是DDPM中非常关键的一步,让模型知道当前的噪声级别
    特征转换：
        通过第二个卷积层进一步处理特征
        最后通过transform层进行空间尺寸变换(上采样或下采样)
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        """
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            time_emb_dim: 时间嵌入的维度
            up: 是否为上采样块
        """
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
            # 上采样模式：
            # 第一个卷积层接收双倍通道数的输入（这是因为UNet中会有跳跃连接）
            # 使用转置卷积(ConvTranspose2d)进行空间尺寸上采样（放大特征图）
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            # 下采样模式：
            # 第一个卷积层接收单倍通道数的输入
            # 使用步长为2的卷积进行空间尺寸下采样（缩小特征图）
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t=None):
        # 第一个卷积层
        h = self.relu(self.bn1(self.conv1(x)))
        
        # 如果有时间嵌入，添加时间信息
        if self.time_mlp and t is not None:
            time_emb = self.relu(self.time_mlp(t))
            time_emb = time_emb[(..., ) + (None, ) * 2]  # 扩展维度以匹配特征图
            h = h + time_emb
            
        # 第二个卷积层和下采样/上采样
        h = self.relu(self.bn2(self.conv2(h)))
        return self.transform(h)


class UNet(nn.Module):
    """
    UNet架构,用于预测添加到图像中的噪声
    包含下采样路径（编码器）和上采样路径（解码器）以及跳跃连接
    """
    def __init__(self, input_channels=3, hidden_dims=64, time_emb_dim=128, num_classes=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 条件嵌入（如果有类别标签）
        self.has_embedding = num_classes is not None
        if self.has_embedding:
            self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        
        # 初始卷积层
        self.conv0 = nn.Conv2d(input_channels, hidden_dims, 3, padding=1)
        
        # 下采样路径
        self.down1 = Block(hidden_dims, hidden_dims, time_emb_dim)
        self.down2 = Block(hidden_dims, 2*hidden_dims, time_emb_dim)
        self.down3 = Block(2*hidden_dims, 4*hidden_dims, time_emb_dim)
        
        # 瓶颈层
        self.bottleneck1 = nn.Conv2d(4*hidden_dims, 4*hidden_dims, 3, padding=1)
        self.bottleneck2 = nn.Conv2d(4*hidden_dims, 4*hidden_dims, 3, padding=1)
        
        # 上采样路径
        self.up1 = Block(2*hidden_dims, 2*hidden_dims, time_emb_dim, up=True)
        self.up2 = Block(2*hidden_dims, hidden_dims, time_emb_dim, up=True)
        self.up3 = Block(hidden_dims, hidden_dims, time_emb_dim, up=True)
        
        # 输出层
        self.output = nn.Conv2d(2*hidden_dims, input_channels, 3, padding=1)
    
    def forward(self, x, t, labels=None):
        # 时间编码
        t = self.time_mlp(t)
        
        # 添加条件嵌入（如果有标签）
        if self.has_embedding and labels is not None:
            t = t + self.label_emb(labels)
        
        # 初始特征提取
        x0 = self.conv0(x)
        
        # 下采样路径
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)
        x3 = self.down3(x2, t)
        
        # 瓶颈层
        x3 = F.relu(self.bottleneck1(x3))
        x3 = F.relu(self.bottleneck2(x3))
        
        # 上采样路径（使用跳跃连接）
        x = self.up1(x3, t)
        x = torch.cat((x, x2), dim=1)  # 跳跃连接
        
        x = self.up2(x, t)
        x = torch.cat((x, x1), dim=1)  # 跳跃连接
        
        x = self.up3(x, t)
        x = torch.cat((x, x0), dim=1)  # 跳跃连接
        
        # 输出预测的噪声
        return self.output(x)


class DDPM(nn.Module):
    """
    去噪扩散概率模型 (DDPM)
    实现前向扩散过程和反向去噪过程
    """
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        """
        初始化DDPM
        
        Args:
            n_steps: 扩散步骤数
            beta_start: 初始噪声调度参数
            beta_end: 最终噪声调度参数
        """
        super().__init__()
        self.n_steps = n_steps
        
        # 设置噪声调度（线性调度）
        self.beta = torch.linspace(beta_start, beta_end, n_steps).to("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def to(self,device):
        """
        将模型移动到指定设备
        
        Args:
            device: 目标设备
        """
        super().to(device)
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self
        
    def add_noise(self, x_0, t):
        """
        向干净图像添加 t 步噪声（前向过程）
        
        Args:
            x_0: 干净图像
            t: 时间步
        
        Returns:
            noisy_image: 添加噪声后的图像
            noise: 添加的噪声
        """
        # 从标准正态分布采样噪声
        noise = torch.randn_like(x_0)
        
        # 为每个图像获取对应的 alpha_bar 值
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        
        # 添加噪声: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy_image = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return noisy_image, noise
    
    def sample(self, model, n_samples, img_size, device, labels=None):
        """
        从模型中采样图像（反向过程）
        
        Args:
            model: UNet模型,用于预测噪声
            n_samples: 要生成的样本数量
            img_size: 图像尺寸
            device: 设备
            labels: 可选的类别标签（用于条件生成）
        
        Returns:
            samples: 生成的图像样本
        """
        # 初始化为纯噪声图像
        channels = model.module.output.out_channels if hasattr(model, "module") else model.output.out_channels
        x_t = torch.randn(n_samples, channels, img_size, img_size).to(device)
        
        # 反向去噪过程
        model.eval()
        with torch.inference_mode():
            for t in tqdm(range(self.n_steps - 1, -1, -1), desc="采样中..."):
                # 为整个批次创建时间张量
                time_tensor = torch.ones(n_samples, device=device, dtype=torch.long) * t
                
                # 预测噪声
                predicted_noise = model(x_t, time_tensor, labels)
                
                # 无噪声系数
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                beta_t = self.beta[t]
                
                # 在t=0时不添加噪声
                if t > 0:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)
                
                # 计算去噪一步的公式
                x_t = 1 / torch.sqrt(alpha_t) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                ) + torch.sqrt(beta_t) * z
        
        # 将像素值缩放到[0,1]范围
        x_t = (x_t.clamp(-1, 1) + 1) / 2
        
        return x_t

    def train_step(self, model, optimizer, x_0, t, labels=None):
        """
        执行一步训练
        
        Args:
            model: UNet模型
            optimizer: 优化器
            x_0: 干净图像
            t: 时间步
            labels: 可选的类别标签（用于条件训练）
        
        Returns:
            loss: 计算的损失值
        """
        # 添加噪声
        noisy_image, target_noise = self.add_noise(x_0, t)
        
        # 预测噪声
        predicted_noise = model(noisy_image, t, labels)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, target_noise)
        
        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()