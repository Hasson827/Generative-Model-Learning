import torch
from torch import optim, nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import Distinguish_Model, Generate_Model
device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.mps.is_available()
                        else "cpu")
print(f"Using device: {device}")

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(root='../data', train=True, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    D = Distinguish_Model().to(device)
    G = Generate_Model().to(device)
    
    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    loss_fn = nn.BCELoss()
    
    epochs = 100
    latent_dim = 128
    
    fixed_noise = torch.randn(16, latent_dim).to(device)
    
    for epoch in range(epochs):
        dis_loss_all = 0 # 判别器损失
        gen_loss_all = 0 # 生成器损失
        loader_len = len(dataloader)
        
        for i, (sample, label) in enumerate(dataloader):
            batch_size = sample.size(0)
            sample = sample.to(device)
            
            # 标签平滑：真实标签使用0.9
            real_label = 0.9
            fake_label = 0.0
            
            # ========= 训练判别器 =========
            D_optim.zero_grad()
            
            # 真实样本
            real_output = D(sample)
            real_loss = loss_fn(real_output,torch.full((batch_size, 1), real_label, device=device))
            
            # 生成假样本
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_sample = G(noise)
            fake_output = D(fake_sample.detach())
            fake_loss = loss_fn(fake_output, torch.full((batch_size, 1), fake_label, device=device))
            
            Dis_loss = real_loss + fake_loss
            Dis_loss.backward()
            D_optim.step()
            
            # ========= 训练生成器 =========
            G_optim.zero_grad()
            
            fake_output = D(fake_sample)
            G_loss = loss_fn(fake_output, torch.full((batch_size, 1), real_label, device=device))
            G_loss.backward()
            G_optim.step()
            
            with torch.inference_mode():
                dis_loss_all += Dis_loss.item()
                gen_loss_all += G_loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Discriminator Loss: {dis_loss_all/loader_len:.4f}, Generator Loss: {gen_loss_all/loader_len:.4f}")
            
            # 可视化生成的样本
            with torch.inference_mode():
                fake_images = G(fixed_noise).view(-1, 1, 28, 28)
                save_images(fake_images, epoch)
    
    return D, G

def save_images(images, epoch):
    """保存生成的图像"""
    images = images.cpu().detach()
    # 反归一化
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    
    plt.figure(figsize=(8, 8))
    for i in range(min(16, images.size(0))):
        plt.subplot(4, 4, i+1)
        if images.dim() == 4:  # 卷积输出
            plt.imshow(images[i].squeeze(), cmap='gray')
        else:  # 线性输出
            plt.imshow(images[i].view(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Generated Images - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'generated_epoch_{epoch}.png')
    plt.close()
        
if __name__ == "__main__":
    D,G = train()
    # 生成最终样本
    with torch.no_grad():
        fake_z = torch.randn(16, 128).to(device)
        fake_sample = G(fake_z)
        fake_sample = fake_sample.cpu().detach()
        fake_sample = (fake_sample + 1) / 2.0  # 反归一化
        fake_sample = torch.clamp(fake_sample, 0, 1)
        
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            if fake_sample.dim() == 4:
                plt.imshow(fake_sample[i].squeeze(), cmap='gray')
            else:
                plt.imshow(fake_sample[i].view(28, 28), cmap='gray')
            plt.axis('off')
        plt.suptitle('Final Generated MNIST Images')
        plt.tight_layout()
        plt.show()