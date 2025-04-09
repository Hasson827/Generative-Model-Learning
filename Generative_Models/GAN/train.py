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
    
    dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    D = Distinguish_Model().to(device)
    G = Generate_Model().to(device)
    
    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    loss_fn = nn.BCELoss()
    
    epochs = 100
    
    for epoch in range(epochs):
        dis_loss_all = 0 # 判别器损失
        gen_loss_all = 0 # 生成器损失
        loader_len = len(dataloader)
        
        for (sample, label) in dataloader:
            # 判别器训练
            sample = sample.view(-1,784).to(device)
            sample_z = torch.randn(sample.size(0), 128).to(device)
            
            Dis_true = D(sample)
            true_loss = loss_fn(Dis_true, torch.ones_like(Dis_true).to(device))
            
            fake_sample = G(sample_z)
            Dis_fake = D(fake_sample.detach())
            fake_loss = loss_fn(Dis_fake, torch.zeros_like(Dis_fake).to(device))
            
            Dis_loss = true_loss + fake_loss

            D_optim.zero_grad()
            Dis_loss.backward()
            D_optim.step()
            
            # 生成器训练
            sample_z = torch.randn(sample.size(0), 128).to(device)
            fake_sample = G(sample_z)
            Dis_G = D(fake_sample)
            G_loss = loss_fn(Dis_G, torch.ones_like(Dis_G).to(device))
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()
            
            with torch.inference_mode():
                dis_loss_all += Dis_loss.item()
                gen_loss_all += G_loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Discriminator Loss: {dis_loss_all/loader_len:.4f}, Generator Loss: {gen_loss_all/loader_len:.4f}")
        
    return D,G
            
if __name__ == "__main__":
    D,G = train()
    fake_z = torch.randn((10, 128)).to(device)
    fake_sample = G(fake_z)
    fake_sample = fake_sample.view(-1, 28, 28)
    fake_sample = fake_sample.cpu().detach().numpy()
    
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(fake_sample[i], cmap='gray')
        plt.axis('off')
    plt.show()