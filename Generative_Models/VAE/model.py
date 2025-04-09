import torch
import torch.nn as nn
import torch.nn.functional as F

# Input img -> Hidden dim -> mean,std -> Reparameterized trick -> Decoder -> Output img
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, h_dim=200, z_dim=20):
        """
        Args:
            input_dim (int): Dimension of the input image
            conditional_dim (int): Dimension of the condition
            h_dim (int): Dimension of the hidden layer
            z_dim (int): Dimension of the latent variable
        """
        super().__init__()
        
        # 保存参数作为实例属性
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        
        # 输入图像的编码器
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim,h_dim),
            nn.ReLU()
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim,h_dim//4),
            nn.ReLU()
        )
        
        # 合并后的处理
        self.combined_encoder = nn.Sequential(
            nn.Linear(h_dim + h_dim//4, h_dim),
            nn.ReLU()
        )
        
        # 均值和方差的线性层
        self.fc_mu = nn.Linear(h_dim,z_dim)
        self.fc_var = nn.Linear(h_dim,z_dim)
        
        # 解码器
        self.decoder_condition_encoder = nn.Sequential(
            nn.Linear(condition_dim,h_dim//4),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + h_dim//4, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
            nn.Sigmoid()  # Sigmoid to ensure output is in [0, 1]
        )
    
    def encode(self,x,c):
        """
        Encode the input image x and condition c into mean and variance
        """
        h_x = self.input_encoder(x)
        h_c = self.condition_encoder(c)
        
        # 合并特征
        h_combined = torch.cat([h_x,h_c],dim=1)
        h = self.combined_encoder(h_combined)
        
        # 均值和方差
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self,mu,log_var):
        """
        Reparameterization trick to sample from the distribution
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self,z,c):
        """
        Decode the latent variable z and condition c into the output image
        """
        # 处理条件信息
        h_c = self.decoder_condition_encoder(c)
        
        # 合并潜在变量和条件信息
        decoder_input = torch.cat([z,h_c],dim=1)
        return self.decoder(decoder_input)
    
    def forward(self,x,c):
        """
        Forward pass through the VAE
        """
        mu, log_var = self.encode(x,c)
        z = self.reparameterize(mu,log_var)
        x_reconstructed = self.decode(z,c)
        return x_reconstructed, mu, log_var
    
    def loss_function(self,x,x_reconstructed,mu,log_var):
        """
        Compute the loss function
        """
        # Binary Cross Entropy loss
        x_reconstructed = torch.clamp(x_reconstructed, min=1e-6, max=1-1e-6)  # Avoid log(0)
        BCE = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        # KL Divergence loss
        var = torch.exp(log_var.clamp(min=-7,max=7))
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - var)
        return BCE + KLD
    
    def sample(self,num_samples,c):
        """
        Sample from the VAE
        """
        z = torch.randn(num_samples, self.z_dim).to(c.device)
        return self.decode(z,c)