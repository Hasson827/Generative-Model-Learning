import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口，包含激活函数、损失函数等
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示工具，用于可视化采样过程


# U-Net组件：残差模块、下采样和上采样模块
class ResidualBlock(nn.Module):
    """
    残差块 - U-Net的基本构建块
    包含时间嵌入和残差连接以帮助信息流动和梯度传播
    
    残差连接是ResNet中的关键创新,通过让网络学习残差映射而非直接映射,
    有效缓解了深度网络中的梯度消失问题，使得训练更加稳定高效
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        """
        初始化残差块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            time_emb_dim (int): 时间嵌入维度，用于条件生成
        """
        super().__init__()
        # 时间嵌入层，将时间信息注入到特征中
        # 扩散模型的关键创新点，使网络能够根据不同的噪声级别(时间步)调整其行为
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 第一个卷积块：卷积+归一化+激活
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)  # 3x3卷积, 保持空间维度
        self.norm1 = nn.GroupNorm(8, out_channels)  # 组归一化，相比BN更适合小批量和生成任务
        
        # 第二个卷积块：卷积+归一化+激活
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # 残差连接通路：如果输入输出通道不匹配，使用1x1卷积调整维度
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)  # 1x1卷积用于调整通道数
        else:
            self.shortcut = nn.Identity()  # 恒等映射，不做任何处理

    def forward(self, x, t):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征图 [批量大小, 通道数, 高度, 宽度]
            t (torch.Tensor): 时间嵌入 [批量大小, time_emb_dim]
            
        返回:
            torch.Tensor: 处理后的特征图 [批量大小, out_channels, 高度, 宽度]
        """
        # 主路径第一阶段处理
        h = self.conv1(x)  # 卷积操作
        h = self.norm1(h)  # 归一化，提高训练稳定性
        h = F.silu(h)  # SiLU激活函数 (Swish: x * sigmoid(x))，性能通常优于ReLU
        
        # 添加时间嵌入 - 关键步骤，使网络能够感知扩散过程的时间步
        time_emb = self.time_mlp(t)  # 将时间嵌入映射到合适的维度 [B, out_channels]
        # 调整形状以便与特征图相加 [B, out_channels, 1, 1]
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)
        # 通过加法融合时间信息，使特征对时间步敏感
        h = h + time_emb
        
        # 主路径第二阶段处理
        h = self.conv2(h)  # 第二次卷积
        h = self.norm2(h)  # 第二次归一化
        h = F.silu(h)  # 第二次激活
        
        # 残差连接：将输入（经过可能的通道调整）与主路径输出相加
        # 这使得信息可以直接流过网络，缓解梯度消失问题
        return h + self.shortcut(x)


class DownSample(nn.Module):
    """
    下采样模块 - 使用步长为2的卷积减小特征图尺寸
    
    在U-Net架构中,下采样用于逐步减小空间维度并增加通道数,
    从而在编码过程中捕获更抽象和语义级的特征
    """
    def __init__(self, channels):
        """
        初始化下采样模块
        
        参数:
            channels (int): 通道数（输入=输出）
        """
        super().__init__()
        # 步长为2的3x3卷积，将特征图尺寸减半
        # 相比最大池化，有参数的下采样可以学习到更好的表示
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            torch.Tensor: 下采样后的特征图 [B, C, H/2, W/2]
        """
        # 应用步长为2的卷积，空间维度减半
        return self.conv(x)


class UpSample(nn.Module):
    """
    上采样模块 - 使用转置卷积增大特征图尺寸
    
    在U-Net架构的解码器部分,上采样用于恢复空间分辨率,
    配合跳跃连接还原高分辨率的特征图
    """
    def __init__(self, channels):
        """
        初始化上采样模块
        
        参数:
            channels (int): 通道数（输入=输出）
        """
        super().__init__()
        # 步长为2的4x4转置卷积，将特征图尺寸加倍
        # 转置卷积（反卷积）可以学习上采样的最佳方式，而不是简单的插值
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            torch.Tensor: 上采样后的特征图 [B, C, H*2, W*2]
        """
        # 应用转置卷积，空间维度加倍
        return self.conv(x)


# 时间嵌入模块
class TimeEmbedding(nn.Module):
    """
    时间步嵌入模块 - 将标量时间步转换为高维特征向量
    
    使用正弦位置编码（类似Transformer）再通过MLP处理，
    这使得模型能够区分不同时间步的噪声水平，是条件生成的关键
    """
    def __init__(self, time_dim):
        """
        初始化时间嵌入模块
        
        参数:
            time_dim (int): 输出时间嵌入的维度
        """
        super().__init__()
        self.time_dim = time_dim
        
        # MLP网络：将初始位置编码映射到高维特征空间
        # 通过非线性变换增强表达能力
        self.mlp = nn.Sequential(
            nn.Linear(time_dim // 4, time_dim),  # 扩展维度
            nn.SiLU(),  # 非线性激活函数
            nn.Linear(time_dim, time_dim)  # 最终映射
        )

    def forward(self, t):
        """
        前向传播 - 将时间步索引转换为嵌入向量
        
        参数:
            t (torch.Tensor): 时间步索引 [batch_size]，范围通常为[0, num_timesteps-1]
            
        返回:
            torch.Tensor: 时间嵌入向量 [batch_size, time_dim]
        """
        # 计算正弦位置编码，使用多个不同频率的正弦波表示位置
        half_dim = self.time_dim // 8  # 使用time_dim的1/8维度来计算位置编码
        
        # 创建指数递减的频率因子
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 计算不同频率: 10000^(-i/d)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        
        # 将时间与频率相乘以获得位置编码的输入
        emb = t[:, None] * emb[None, :]  # [batch_size, half_dim]，通过广播实现
        
        # 使用正弦和余弦函数增强位置表示的区分性
        # [batch_size, half_dim*2]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # MLP处理得到最终的时间嵌入
        # 非线性变换增强表达能力，得到更丰富的时间表示
        emb = self.mlp(emb)  # [batch_size, time_dim]
        return emb


# U-Net模型
class UNet(nn.Module):
    """
    U-Net模型 - 扩散模型的骨干网络
    
    U-Net是一种编码器-解码器架构，带有跳跃连接，
    最初为医学图像分割设计，在扩散模型中用于预测噪声。
    该结构能够同时捕获全局语义信息和局部细节
    """
    def __init__(
        self, 
        in_channels=3,      # 输入通道数（通常是RGB图像，3通道）
        out_channels=3,     # 输出通道数（预测的噪声，通常与输入通道相同）
        time_dim=256,       # 时间嵌入维度
        base_channels=64,   # 基础通道数（第一层）
        channel_mults=(1, 2, 4, 8)  # 通道数的倍增因子
    ):
        """
        初始化U-Net模型
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            time_dim (int): 时间嵌入维度
            base_channels (int): 基础通道数（第一层）
            channel_mults (tuple): 每层通道数的倍增因子
        """
        super().__init__()
        # 时间嵌入层 - 将时间步索引转换为嵌入向量
        self.time_embedding = TimeEmbedding(time_dim)
        
        # 初始卷积层 - 将图像映射到特征空间
        # 3x3卷积，保持空间尺寸不变
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 构建下采样路径（编码器）
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        # 保存下采样路径的通道数，后续用于跳跃连接
        down_channels = []
        
        # 按照通道倍增因子创建下采样blocks
        for mult in channel_mults:
            out_channels_mult = base_channels * mult  # 计算当前层的输出通道数
            
            # 添加残差块
            self.down_blocks.append(ResidualBlock(current_channels, out_channels_mult, time_dim))
            current_channels = out_channels_mult
            down_channels.append(current_channels)  # 记录通道数用于后续跳跃连接
            
            # 添加下采样模块（最后一层除外）
            if mult != channel_mults[-1]:
                self.down_blocks.append(DownSample(current_channels))
                down_channels.append(current_channels)  # 下采样后通道数不变
        
        # 中间块 - U-Net最底部
        # 连接编码器和解码器，处理最低分辨率的特征
        self.middle_block = ResidualBlock(current_channels, current_channels, time_dim)
        
        # 构建上采样路径（解码器）
        self.up_blocks = nn.ModuleList()
        
        # 逆序遍历通道倍增因子来构建解码器
        for i, mult in enumerate(reversed(channel_mults)):
            out_channels_mult = base_channels * mult
            
            # 上采样路径中每个残差块需要结合编码器对应层的特征
            # 因此输入通道数是当前通道数加上跳跃连接的通道数
            self.up_blocks.append(
                ResidualBlock(current_channels + down_channels.pop(), out_channels_mult, time_dim)
            )
            current_channels = out_channels_mult
            
            # 添加上采样模块（最后一层除外）
            if i < len(channel_mults) - 1:
                self.up_blocks.append(UpSample(current_channels))
        
        # 最终输出层，将特征映射回图像空间
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, current_channels),  # 归一化，提高训练稳定性
            nn.SiLU(),  # 非线性激活函数
            nn.Conv2d(current_channels, out_channels, 3, padding=1)  # 3x3卷积得到最终输出
        )
    
    def forward(self, x, t):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入噪声图像 [B, in_channels, H, W]
            t (torch.Tensor): 时间步索引 [B]
            
        返回:
            torch.Tensor: 预测的噪声 [B, out_channels, H, W]
        """
        # 时间嵌入处理 - 将标量时间步转换为向量表示
        time_emb = self.time_embedding(t)  # [B, time_dim]
        
        # 初始特征提取
        h = self.initial_conv(x)  # [B, base_channels, H, W]
        
        # 保存下采样路径的特征用于后续跳跃连接
        skip_connections = []
        
        # 下采样路径 - 编码器
        # 逐步减小空间分辨率，增加通道数，提取更抽象的特征
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                # 残差块处理，带有时间条件
                h = block(h, time_emb)
            else:
                # 下采样操作，减小空间尺寸
                h = block(h)
            # 保存每个层的输出用于后续的跳跃连接
            skip_connections.append(h)
        
        # 中间块 - U-Net底部，处理最低分辨率特征
        h = self.middle_block(h, time_emb)
        
        # 上采样路径 - 解码器
        # 逐步恢复空间分辨率，利用跳跃连接保留细节
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                # 从编码器获取对应层的特征进行拼接
                # skip_connections以后进先出(LIFO)的方式使用
                skip = skip_connections.pop()  # 获取对应层的跳跃连接
                h = torch.cat([h, skip], dim=1)  # 在通道维度上拼接特征
                # 残差块处理，融合特征
                h = block(h, time_emb)
            else:
                # 上采样操作，增大空间尺寸
                h = block(h)
        
        # 最终输出层 - 生成预测的噪声
        return self.final_conv(h)  # [B, out_channels, H, W]


# DDPM扩散模型
class DDPM(nn.Module):
    """
    去噪扩散概率模型(DDPM) - 一种基于扩散过程的生成模型
    
    DDPM的核心思想是定义一个前向扩散过程逐渐将图像变为噪声，
    然后训练一个神经网络来学习反向过程，即从噪声中恢复图像。
    
    参考论文: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """
    def __init__(
        self, 
        model,                # 噪声预测模型（通常是U-Net）
        beta_start=1e-4,      # β调度的起始值
        beta_end=0.02,        # β调度的结束值
        num_timesteps=1000,   # 扩散过程的总时间步数
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # 计算设备
    ):
        """
        初始化DDPM模型
        
        参数:
            model (nn.Module): 噪声预测网络
            beta_start (float): 噪声调度起始β值
            beta_end (float): 噪声调度终止β值
            num_timesteps (int): 扩散过程总时间步数
            device (str): 计算设备
        """
        super().__init__()
        self.model = model  # 噪声预测网络
        self.num_timesteps = num_timesteps  # 总时间步数
        self.device = device  # 计算设备
        
        # 创建线性噪声调度 - β从beta_start线性增加到beta_end
        # β调度控制每一步添加噪声的比例
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        
        # α值 = 1-β，表示保留原始信号的比例
        self.alphas = 1. - self.betas
        
        # 累积α值 - α_hat_t = α_1 * α_2 * ... * α_t
        # 表示t步后原始信号保留的比例
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算扩散过程中需要的各种常数，避免重复计算以提高效率
        # 这些值在前向扩散和反向扩散过程中都会用到
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # √α_hat
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)  # √(1-α_hat)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)  # log(1-α_hat)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)  # 1/√α
        
        # 修复后验方差计算
        # 原来的代码有问题: self.posterior_variance = self.betas * (1. - self.alphas_cumprod.previous(0)) / (1. - self.alphas_cumprod)
        # 修复：计算正确的后验方差
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod[:-1]) / (1. - self.alphas_cumprod[1:])
        # 为t=0的情况添加一个值
        self.posterior_variance = torch.cat([torch.tensor([self.betas[0]], device=device), self.posterior_variance])
        
        # 计算后验均值系数
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_cumprod[1:]) * self.betas / (1. - self.alphas_cumprod[1:])
        self.posterior_mean_coef2 = torch.sqrt(self.alphas[1:]) * (1. - self.alphas_cumprod[:-1]) / (1. - self.alphas_cumprod[1:])
        # 为t=0的情况添加占位符
        self.posterior_mean_coef1 = torch.cat([torch.tensor([0.0], device=device), self.posterior_mean_coef1])
        self.posterior_mean_coef2 = torch.cat([torch.tensor([1.0], device=device), self.posterior_mean_coef2])
        
    def get_index_from_list(self, vals, t, x_shape):
        """
        从预计算的张量中获取指定时间步的值，并调整形状以便于广播操作
        
        参数:
            vals (torch.Tensor): 预计算的张量 [num_timesteps]
            t (torch.Tensor): 时间步索引 [batch_size]
            x_shape (tuple): 输入特征的形状
            
        返回:
            torch.Tensor: 形状调整后的值，便于与图像张量进行广播操作
        """
        batch_size = t.shape[0]
        # 根据时间步索引t从vals中获取对应值
        # gather操作从vals的最后一维中按照索引t选择元素
        out = vals.gather(-1, t)
        
        # 调整形状以便与图像张量进行广播操作
        # 例如，如果x_shape是[B,C,H,W]，则返回形状为[B,1,1,1]
        # 这样可以直接与图像张量进行按位乘法
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def forward_diffusion(self, x_0, t):
        """
        前向扩散过程 q(x_t | x_0)
        逐步将原始图像添加噪声，直到完全变为随机噪声
        
        参数:
            x_0 (torch.Tensor): 原始图像 [B, C, H, W]
            t (torch.Tensor): 目标时间步索引 [B]
            
        返回:
            tuple: (噪声图像x_t, 添加的噪声)
        """
        # 生成随机噪声，与x_0形状相同
        noise = torch.randn_like(x_0)
        
        # 获取时间步t对应的√α_hat_t
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        # 获取时间步t对应的√(1-α_hat_t)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # 构建噪声图像x_t：闭式解
        # x_t = √α_hat_t * x_0 + √(1-α_hat_t) * ε，其中ε是标准正态噪声
        # 这个公式是q(x_t|x_0)分布的重参数化形式
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        # 返回噪声图像和添加的噪声
        return x_t, noise
    
    def training_loss(self, x_0):
        """
        计算训练损失 - 简单噪声预测损失
        
        DDPM的训练目标是让神经网络预测在给定时间步t的噪声图像x_t中包含的噪声
        
        参数:
            x_0 (torch.Tensor): 原始图像批次 [B, C, H, W]
            
        返回:
            torch.Tensor: 损失值
        """
        batch_size = x_0.shape[0]
        
        # 随机采样时间步 - 每个图像随机采样一个时间步
        # 这样确保模型能够学习到扩散过程的每一个阶段
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # 应用前向扩散得到噪声图像x_t和添加的噪声
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # 使用模型预测x_t中包含的噪声
        predicted_noise = self.model(x_t, t)
        
        # 计算简单MSE损失 - 预测噪声与实际添加噪声之间的均方误差
        # 这是DDPM论文中的简化目标，实际上是变分下界的简化形式
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, image_size, batch_size=1, channels=3):
        """
        从纯噪声采样生成图像 - 反向扩散过程
        
        通过迭代反向扩散步骤，逐步将随机噪声转换为真实图像
        
        参数:
            image_size (int): 生成图像的尺寸
            batch_size (int): 批量大小
            channels (int): 图像通道数
            
        返回:
            torch.Tensor: 生成的图像 [B, C, H, W]
        """
        # 设置为评估模式
        self.model.eval()
        
        # 从标准正态分布采样纯噪声作为起点(x_T)
        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        # 逐步反向扩散过程: 从x_T到x_0
        # 使用tqdm显示采样进度
        for t in tqdm(reversed(range(self.num_timesteps)), desc="采样进度"):
            # 创建时间步张量，所有样本使用相同的时间步t
            time_tensor = torch.tensor([t] * batch_size, device=self.device).long()
            
            # 使用模型预测噪声
            predicted_noise = self.model(x, time_tensor)
            
            # 获取当前时间步的参数
            alpha = self.alphas[t]  # α_t
            alpha_cumprod = self.alphas_cumprod[t]  # α_hat_t
            beta = self.betas[t]  # β_t
            
            # 根据时间步决定是否添加随机噪声
            # 在最后一步(t=0)不添加噪声，以获得确定性的结果
            if t > 0:
                # 添加噪声 - 除了最后一步
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            # 反向扩散步骤 - 从x_t计算x_{t-1}
            # 使用预测的噪声更新样本
            # 公式来自DDPM论文，是p_θ(x_{t-1}|x_t)的均值和方差参数化
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
        # 恢复训练模式
        self.model.train()
        return x
    
    def train_step(self, optimizer, x_0):
        """
        执行一步训练
        
        封装了损失计算、反向传播和优化器更新的完整训练步骤
        
        参数:
            optimizer (torch.optim.Optimizer): PyTorch优化器
            x_0 (torch.Tensor): 训练批次数据 [B, C, H, W]
            
        返回:
            torch.Tensor: 当前批次的损失值
        """
        # 清零梯度，避免梯度累积
        optimizer.zero_grad()
        
        # 计算损失
        loss = self.training_loss(x_0)
        
        # 反向传播：计算参数梯度
        loss.backward()
        
        # 优化器步进：更新模型参数
        optimizer.step()
        
        return


# 使用示例
def create_ddpm_model(
    image_size=32,  # 图像尺寸
    in_channels=3,  # 输入通道数
    out_channels=3,  # 输出通道数
    base_channels=64,  # 基础通道数
    channel_mults=(1, 2, 4, 8),  # 通道倍增因子
    time_dim=256  # 时间嵌入维度
):
    """
    创建DDPM模型的辅助函数
    
    参数:
        image_size: 图像尺寸
        in_channels: 输入通道数
        out_channels: 输出通道数
        base_channels: 基础通道数
        channel_mults: 通道倍增因子
        time_dim: 时间嵌入维度
        
    返回:
        配置好的DDPM模型实例
    """
    # 创建U-Net模型作为噪声预测器
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        base_channels=base_channels,
        channel_mults=channel_mults
    )
    
    # 创建DDPM模型实例
    ddpm = DDPM(model=model)
    return ddpm
