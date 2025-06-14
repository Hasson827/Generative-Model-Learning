import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Distinguish_Model, Generate_Model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.mps.is_available()
                      else "cpu")

print(f"Using device: {device}")
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(root="../data", train=True, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    D = Distinguish_Model().to(device)
    G = Generate_Model().to(device)
    
    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    loss_fn = nn.BCELoss()
    
    epochs = 100
    latent_dim = 128
    
    fixed_noise = torch.randn(10, latent_dim, device=device)
    fixed_labels = torch.arange(0, 10, device=device)
    
    for epoch in range(epochs):
        dis_loss_all = 0
        gen_loss_all = 0
        loader_len = len(dataloader)
        
        for i, (sample, real_labels) in enumerate(dataloader):
            batch_size = sample.size(0)
            sample = sample.to(device)
            real_labels = real_labels.to(device)
            
            real_label_smooth = 0.9
            fake_label_smooth = 0.0
            
            # 训练判别器
            D_optim.zero_grad()
            
            # 真实样本，使用真实标签
            real_output = D(sample, real_labels)
            real_loss = loss_fn(real_output, torch.full((batch_size, 1), real_label_smooth, device=device))
            
            # 生成假样本，随机生成标签
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_labels = torch.randint(0,10,(batch_size,),device=device)
            fake_sample = G(noise, fake_labels)
            fake_output = D(fake_sample.detach(), fake_labels)
            fake_loss = loss_fn(fake_output, torch.full((batch_size, 1), fake_label_smooth, device=device))
            
            dis_loss = real_loss + fake_loss
            dis_loss.backward()
            D_optim.step()
            
            # 训练生成器
            G_optim.zero_grad()
            
            # 生成器希望判别器认为生成的样本是真的
            fake_output = D(fake_sample, fake_labels)
            gen_loss = loss_fn(fake_output, torch.full((batch_size, 1), real_label_smooth, device=device))
            gen_loss.backward()
            G_optim.step()
            
            with torch.inference_mode():
                dis_loss_all += dis_loss.item()
                gen_loss_all += gen_loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Discriminator Loss: {dis_loss_all/loader_len:.4f}, Generator Loss: {gen_loss_all/loader_len:.4f}")
            
            with torch.inference_mode():
                fake_images = G(fixed_noise, fixed_labels)
                save_conditional_images(fake_images, fixed_labels, epoch)
    
    return D, G

def save_conditional_images(images, labels, epoch):
    """保存条件生成的图像"""
    images = images.cpu().detach()
    labels = labels.cpu().detach()
    
    # 反归一化
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.suptitle(f'Conditional Generated Images - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'conditional_generated_epoch_{epoch}.png')
    plt.close()

def generate_specific_digits(G, device, target_digits, num_samples=5):
    """生成指定数字的样本"""
    G.eval()
    with torch.inference_mode():
        plt.figure(figsize=(15, len(target_digits) * 3))
        
        for row, digit in enumerate(target_digits):
            # 为每个数字生成多个样本
            noise = torch.randn(num_samples, 128).to(device)
            labels = torch.full((num_samples,), digit).to(device)
            
            generated_images = G(noise, labels)
            generated_images = (generated_images + 1) / 2.0  # 反归一化
            generated_images = torch.clamp(generated_images, 0, 1)
            
            for col in range(num_samples):
                plt.subplot(len(target_digits), num_samples, row * num_samples + col + 1)
                plt.imshow(generated_images[col].cpu().squeeze(), cmap='gray')
                if col == 0:
                    plt.ylabel(f'Digit {digit}', fontsize=12)
                plt.axis('off')
        
        plt.suptitle('Generated Specific Digits', fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    D, G = train()
    
    # 测试条件生成
    print("Generating specific digits...")
    
    # 生成指定数字
    target_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    generate_specific_digits(G, device, target_digits, num_samples=5)
    
    # 交互式生成
    while True:
        try:
            digit = int(input("Enter a digit (0-9) to generate, or -1 to exit: "))
            if digit == -1:
                break
            if 0 <= digit <= 9:
                generate_specific_digits(G, device, [digit], num_samples=8)
            else:
                print("Please enter a digit between 0 and 9")
        except ValueError:
            print("Please enter a valid integer")
        except KeyboardInterrupt:
            break