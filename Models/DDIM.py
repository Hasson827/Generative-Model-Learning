import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口，包含激活函数、损失函数等
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示工具，用于可视化采样进度
from typing import List, Optional, Tuple, Union  # 类型提示，增强代码可读性


class ResidualBlock(nn.Module):
    """
    残差块 - U-Net模型的基本构建块
    包含时间嵌入条件和残差连接
    
    残差连接的核心思想是让网络学习残差映射而不是直接映射，
    有助于解决深层网络的梯度消失问题，并提高训练稳定性
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        """
        初始化残差块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            time_emb_dim (int): 时间嵌入的维度，用于条件生成
        """
        super().__init__()
        # 时间嵌入映射层，将时间信息注入到特征中
        # 这是扩散模型的关键部分，使网络能够针对不同的噪声水平(时间步)调整其行为
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 第一个卷积块：卷积+归一化+激活
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)  # 3x3卷积，保持空间尺寸
        self.norm1 = nn.GroupNorm(8, out_channels)  # 组归一化，比BN更适合小批量情况
        
        # 第二个卷积块：卷积+归一化+激活
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # 残差连接路径：如果输入输出通道不匹配，使用1x1卷积进行调整
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)  # 1x1卷积用于调整通道数
        else:
            self.shortcut = nn.Identity()  # 通道数相同时直接使用恒等映射

    def forward(self, x, t):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征图 [批量大小, 通道数, 高度, 宽度]
            t (torch.Tensor): 时间嵌入 [批量大小, 时间嵌入维度]
            
        返回:
            torch.Tensor: 处理后的特征图 [批量大小, 输出通道数, 高度, 宽度]
        """
        # 主路径第一阶段
        h = self.conv1(x)  # 卷积操作
        h = self.norm1(h)  # 归一化
        h = F.silu(h)  # SiLU激活函数 (Swish: x * sigmoid(x))，比ReLU效果更好
        
        # 添加时间嵌入信息 - 扩散模型的关键步骤
        time_emb = self.time_mlp(t)  # 将时间嵌入映射到合适的维度 [B, out_channels]
        # 调整形状以便于特征图相加 [B, out_channels, 1, 1]
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)
        h = h + time_emb  # 时间信息作为加性条件，让特征对时间步敏感
        
        # 主路径第二阶段
        h = self.conv2(h)  # 第二次卷积
        h = self.norm2(h)  # 第二次归一化
        h = F.silu(h)  # 第二次激活
        
        # 残差连接：主路径输出加上捷径连接
        # 这使得信息和梯度可以直接流过网络，有助于训练更深的模型
        return h + self.shortcut(x)


class DownSample(nn.Module):
    """
    下采样模块 - 使用步长为2的卷积减半特征图尺寸
    
    在U-Net架构中用于减小特征图的空间尺寸，增加通道数，
    从而在编码过程中逐渐提取更抽象的特征
    """
    def __init__(self, channels):
        """
        初始化下采样模块
        
        参数:
            channels (int): 通道数（输入=输出）
        """
        super().__init__()
        # 使用步长为2的3x3卷积代替池化操作，可以学习下采样过程
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
    
    在U-Net架构的解码器部分用于恢复特征图的空间尺寸，
    配合跳跃连接一起重建高分辨率特征
    """
    def __init__(self, channels):
        """
        初始化上采样模块
        
        参数:
            channels (int): 通道数（输入=输出）
        """
        super().__init__()
        # 使用转置卷积(反卷积)进行上采样，学习上采样的权重
        # 4x4核大小，步长为2，填充为1，可以精确地将特征图尺寸加倍
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

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


class TimeEmbedding(nn.Module):
    """
    时间嵌入模块 - 将标量时间步转换为高维特征向量
    
    使用正弦位置编码(类似Transformer)再通过MLP处理，
    这使得模型能够区分不同的噪声水平(时间步)，是条件生成的关键
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
        # 首先将时间编码扩展，然后进行非线性变换
        self.mlp = nn.Sequential(
            nn.Linear(time_dim // 4, time_dim),  # 第一层：扩展维度
            nn.SiLU(),  # 非线性激活函数
            nn.Linear(time_dim, time_dim)  # 第二层：最终映射
        )

    def forward(self, t):
        """
        前向传播 - 将时间步索引转换为嵌入向量
        
        参数:
            t (torch.Tensor): 时间步索引 [batch_size]，范围通常是[0, num_timesteps-1]
            
        返回:
            torch.Tensor: 时间嵌入向量 [batch_size, time_dim]
        """
        # 计算正弦位置编码，类似于Transformer中的位置编码
        # 使用不同频率的正弦波来表示位置信息
        half_dim = self.time_dim // 8  # 使用time_dim的1/8维度来计算位置编码
        
        # 创建指数递减的对数空间因子
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 计算频率因子: 10000^(-i/d)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        
        # 将时间t与不同频率相乘，得到位置编码的输入
        # [batch_size, half_dim]
        emb = t[:, None] * emb[None, :]  # 广播操作
        
        # 对时间进行正弦和余弦编码，获得更丰富的表示
        # [batch_size, half_dim*2]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # 通过MLP网络处理位置编码，得到最终的时间嵌入
        # [batch_size, time_dim]
        emb = self.mlp(emb)
        return emb


class UNet(nn.Module):
    """
    U-Net网络 - DDIM的骨干网络，负责预测噪声
    
    U-Net是一种编码器-解码器架构，具有跳跃连接，
    专为图像生成/分割设计，这里用于从噪声图像中预测噪声
    """
    def __init__(
        self, 
        in_channels=3,      # 输入通道数（通常是RGB图像，3通道）
        out_channels=3,     # 输出通道数（预测的噪声，通常与输入相同）
        time_dim=256,       # 时间嵌入维度
        base_channels=64,   # 基础通道数（第一层）
        channel_mults=(1, 2, 4, 8)  # 各层通道数的倍增因子
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
        
        # 时间嵌入层 - 将时间步转换为嵌入向量
        self.time_embedding = TimeEmbedding(time_dim)
        
        # 初始卷积层 - 将图像映射到特征空间
        # 3x3卷积，不改变空间尺寸
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 构建下采样路径（编码器）
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        # 存储每层的输出通道数，用于后续跳跃连接
        down_channels = []
        
        # 按照通道倍增因子构建下采样路径
        for mult in channel_mults:
            out_channels = base_channels * mult  # 计算当前层的输出通道数
            
            # 添加残差块
            self.down_blocks.append(ResidualBlock(current_channels, out_channels, time_dim))
            current_channels = out_channels
            down_channels.append(current_channels)  # 记录该层通道数用于跳跃连接
            
            # 每个分辨率级别后添加下采样层（最后一层除外）
            if mult != channel_mults[-1]:
                self.down_blocks.append(DownSample(current_channels))
                down_channels.append(current_channels)  # 下采样后通道数不变
        
        # 中间块 - U-Net底部
        # 在编码器和解码器之间起到桥梁作用
        self.middle_block = ResidualBlock(current_channels, current_channels, time_dim)
        
        # 构建上采样路径（解码器）
        self.up_blocks = nn.ModuleList()
        
        # 逆序遍历通道倍增因子来构建解码器
        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            
            # 上采样路径中每个残差块会结合编码器对应层的跳跃连接
            # 因此输入通道数是当前通道数加上跳跃连接的通道数
            self.up_blocks.append(
                ResidualBlock(current_channels + down_channels.pop(), out_channels, time_dim)
            )
            current_channels = out_channels
            
            # 每个分辨率级别后添加上采样层（最后一层除外）
            if i < len(channel_mults) - 1:
                self.up_blocks.append(UpSample(current_channels))
        
        # 最终输出层 - 将特征映射回图像空间
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, current_channels),  # 归一化
            nn.SiLU(),  # 激活函数
            # 3x3卷积得到最终输出，通道数等于输入图像的通道数
            nn.Conv2d(current_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入噪声图像 [B, C, H, W]
            t (torch.Tensor): 时间步索引 [B]
            
        返回:
            torch.Tensor: 预测的噪声 [B, C, H, W]
        """
        # 处理时间嵌入
        time_emb = self.time_embedding(t)  # [B, time_dim]
        
        # 初始卷积，将输入映射到特征空间
        h = self.initial_conv(x)  # [B, base_channels, H, W]
        
        # 保存下采样路径上的每层输出，用于后续跳跃连接
        skip_connections = []
        
        # 下采样路径（编码器）
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                # 残差块处理，包含时间条件
                h = block(h, time_emb)
            else:
                # 下采样操作
                h = block(h)
            # 保存每个层的输出用于跳跃连接
            skip_connections.append(h)
        
        # 中间块处理 - U-Net的底部
        h = self.middle_block(h, time_emb)
        
        # 上采样路径（解码器）
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                # 获取并连接对应的跳跃连接
                # skip_connections是后进先出(LIFO)，与上采样路径层级匹配
                skip = skip_connections.pop()
                # 在通道维度上拼接特征
                h = torch.cat([h, skip], dim=1)
                # 残差块处理
                h = block(h, time_emb)
            else:
                # 上采样操作
                h = block(h)
        
        # 最终输出层 - 得到预测的噪声
        return self.final_conv(h)


class DDIM(nn.Module):
    """
    去噪扩散隐式模型(DDIM) - DDPM的确定性变体
    
    DDIM改进了DDPM，使用确定性采样过程加速生成，
    并允许在随机性与确定性之间平滑调整
    
    核心改进：
    1. 确定性采样过程，可以跳过时间步
    2. 引入参数eta控制随机性
    3. 更高效的推理过程，需要更少的采样步骤
    """
    def __init__(
        self, 
        model,  # 噪声预测网络
        beta_start=1e-4,  # β调度的起始值
        beta_end=0.02,    # β调度的结束值
        num_timesteps=1000,  # 总时间步数
        device="cuda" if torch.cuda.is_available() else "cpu",  # 计算设备
        eta=0.0  # η参数控制随机性，η=0时为完全确定性，η=1时退化为DDPM
    ):
        """
        初始化DDIM模型
        
        参数:
            model (nn.Module): 噪声预测网络(通常是U-Net)
            beta_start (float): 噪声调度起始β值
            beta_end (float): 噪声调度终止β值
            num_timesteps (int): 扩散过程的总时间步数
            device (str): 计算设备
            eta (float): 随机性控制参数，0为确定性，1为完全随机(DDPM)
        """
        super().__init__()
        self.model = model  # 噪声预测网络
        self.num_timesteps = num_timesteps  # 总时间步数
        self.device = device  # 计算设备
        self.eta = eta  # DDIM特有参数，控制随机性水平
        
        # 构建线性噪声调度 - β从beta_start线性增加到beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        # α值 = 1-β
        self.alphas = 1. - self.betas
        # 累积α值 - α_hat = α_1 * α_2 * ... * α_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算扩散过程中需要的常数，避免重复计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # √α_hat
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)  # √(1-α_hat)
        
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
        out = vals.gather(-1, t)
        # 调整形状以便于与图像张量进行广播操作 [batch_size, 1, 1, 1]
        # 例如可以直接与形状为[batch_size, channels, height, width]的图像相乘
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def forward_diffusion(self, x_0, t):
        """
        前向扩散过程 q(x_t | x_0)
        与DDPM相同，将干净图像逐步添加噪声
        
        参数:
            x_0 (torch.Tensor): 原始图像 [B, C, H, W]
            t (torch.Tensor): 时间步索引 [B]
            
        返回:
            tuple: (噪声图像x_t, 添加的噪声)
        """
        # 生成随机噪声，与x_0形状相同
        noise = torch.randn_like(x_0)
        
        # 获取时间步t对应的α_hat值的平方根
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        # 获取时间步t对应的(1-α_hat)的平方根
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # 计算噪声图像x_t
        # 公式: x_t = √α_hat_t * x_0 + √(1-α_hat_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def training_loss(self, x_0):
        """
        计算训练损失 - 与DDPM相同
        DDIM的训练目标与DDPM相同，区别在于采样过程
        
        参数:
            x_0 (torch.Tensor): 原始图像 [B, C, H, W]
            
        返回:
            torch.Tensor: 损失值
        """
        batch_size = x_0.shape[0]
        
        # 随机采样时间步 - 每个样本在[0, num_timesteps-1]范围内随机选择一个时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # 应用前向扩散，得到时间步t的噪声图像和添加的噪声
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # 使用模型预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 计算均方误差损失 - 预测噪声与实际噪声之间的MSE
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def predict_x0_from_noise(self, x_t, t, noise):
        """
        从噪声图像x_t和预测的噪声中恢复原始图像x0
        DDIM的关键方法之一，用于构建确定性采样过程
        
        参数:
            x_t (torch.Tensor): 时间步t的噪声图像
            t (torch.Tensor): 时间步索引
            noise (torch.Tensor): 预测的噪声
            
        返回:
            torch.Tensor: 估计的原始图像x0
        """
        # 获取时间步t对应的α_hat值的平方根
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_t.shape)
        # 获取时间步t对应的(1-α_hat)的平方根
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # 使用预测的噪声从x_t逆向恢复x0
        # 公式来自前向过程的重排: x_0 = (x_t - √(1-α_hat_t) * noise) / √α_hat_t
        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        return x0_pred
    
    @torch.no_grad()
    def sample(self, image_size, batch_size=1, channels=3, sampling_timesteps=50):
        """
        使用DDIM采样算法从噪声生成图像
        
        DDIM的核心采样算法，允许以较少的步骤实现高质量采样
        
        参数:
            image_size (int): 生成图像的尺寸
            batch_size (int): 批量大小
            channels (int): 图像通道数
            sampling_timesteps (int): 采样时间步数，可以远小于训练时间步数
            
        返回:
            torch.Tensor: 生成的图像 [B, C, H, W]
        """
        # 设置为评估模式
        self.model.eval()
        
        # 从标准正态分布采样纯噪声作为起点(x_T)
        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        # 计算采样间隔 - DDIM关键改进
        # 这允许我们以更少的步骤进行采样，例如用50步代替1000步
        times = torch.linspace(self.num_timesteps-1, 0, sampling_timesteps+1, device=self.device).long()
        # 将时间步配对，例如[(t_n, t_{n-1}), (t_{n-1}, t_{n-2}), ..., (t_1, t_0)]
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # 反向扩散过程
        for time, time_next in tqdm(time_pairs, desc="DDIM Sampling"):
            # 当前时间步的张量，形状为[batch_size]
            time_tensor = torch.tensor([time] * batch_size, device=self.device).long()
            
            # 使用模型预测噪声
            predicted_noise = self.model(x, time_tensor)
            
            # 获取当前和下一时间步的α_cumprod
            alpha_cumprod = self.alphas_cumprod[time]
            # 对于最后一步，下一个alpha_cumprod是1
            alpha_cumprod_next = self.alphas_cumprod[time_next] if time_next >= 0 else torch.tensor(1.0, device=self.device)
            
            # DDIM算法的核心步骤1：估计x_0
            # 使用当前预测的噪声来估计原始图像
            x0_predicted = self.predict_x0_from_noise(x, time_tensor, predicted_noise)
            
            # DDIM算法的核心步骤2：计算确定性更新的方差
            # η控制随机性，η=0时为确定性DDIM，η=1时等同于DDPM
            sigma_t = self.eta * torch.sqrt(
                (1 - alpha_cumprod_next) / (1 - alpha_cumprod) * 
                (1 - alpha_cumprod / alpha_cumprod_next)
            )
            
            # 根据η参数决定是否添加随机噪声
            noise = torch.zeros_like(x) if self.eta == 0 else torch.randn_like(x)
            
            # DDIM确定性更新公式
            # 公式: x_{t-1} = √α_hat_{t-1} * x_0 + √(1-α_hat_{t-1}-σ_t^2) * ε_θ(x_t, t) + σ_t * z
            x = torch.sqrt(alpha_cumprod_next) * x0_predicted + \
                torch.sqrt(1 - alpha_cumprod_next - sigma_t**2) * predicted_noise + \
                sigma_t * noise
            
        # 恢复训练模式
        self.model.train()
        return x
    
    @torch.no_grad()
    def sample_ddpm_style(self, image_size, batch_size=1, channels=3):
        """
        传统DDPM风格的采样 - 当η=1时，DDIM等同于DDPM
        使用所有时间步的完整采样过程
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        # 反向扩散过程
        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
            time_tensor = torch.tensor([t] * batch_size, device=self.device).long()
            
            # 预测噪声
            predicted_noise = self.model(x, time_tensor)
            
            # DDPM的更新公式
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            # 使用预测的噪声更新样本
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
        self.model.train()
        return x
    
    def train_step(self, optimizer, x_0):
        """执行一步训练"""
        optimizer.zero_grad()
        loss = self.training_loss(x_0)
        loss.backward()
        optimizer.step()
        return loss


def create_ddim_model(
    image_size=32, 
    in_channels=3, 
    out_channels=3,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    time_dim=256,
    eta=0.0  # 默认为完全确定性模式
):
    """
    创建DDIM模型
    eta: 控制随机性的参数，0为完全确定性，1为等同于DDPM
    """
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        base_channels=base_channels,
        channel_mults=channel_mults
    )
    
    ddim = DDIM(model=model, eta=eta)
    return ddim
