import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, h_dim=400, z_dim=20):
        """
        条件VAE模型
        Args:
            input_dim (int): 输入维度
            condition_dim (int): 条件维度(类别的one-hot编码维度)
            h_dim (int): 隐藏层维度
            z_dim (int): 潜在空间维度
        """
        super().__init__()
        
        # 编码器（接收图像和条件）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        
        # 解码器（接收潜在变量和条件）
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + condition_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        """条件编码过程"""
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, c):
        """条件解码过程"""
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, c):
        """前向传播"""
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z, c)
        return x_reconstructed, mu, log_var
    
    def loss_function(self, x, x_reconstructed, mu, log_var):
        """计算VAE损失"""
        recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, c):
        """给定条件c,从潜在空间采样生成新样本"""
        z = torch.randn(num_samples, self.fc_mu.out_features).to(c.device)
        return self.decode(z, c)

if __name__ == "__main__":
    # 测试模型
    input_dim = 784  # 28x28
    condition_dim = 10  # 假设有10个类别
    model = ConditionalVAE(input_dim, condition_dim)
    
    # 随机输入和条件
    x = torch.randn(64, input_dim)  # batch_size=64
    c = torch.randn(64, condition_dim)  # batch_size=64
    
    # 前向传播
    x_reconstructed, mu, log_var = model(x, c)
    sample = model.sample(64, c)
    
    # 打印输出形状
    print("Reconstructed shape:", x_reconstructed.shape)
    print("Mu shape:", mu.shape)
    print("Log variance shape:", log_var.shape)
    print("Sample shape:", sample.shape)