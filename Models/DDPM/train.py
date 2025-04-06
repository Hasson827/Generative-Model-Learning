import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from model import UNet,DDPM

transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.mps.is_available()
                      else "cpu")
model = UNet(input_channels=1,hidden_dims=64).to(device)
ddpm = DDPM(n_steps=1000).to(device)

lr = 1e-4
num_epochs = 100

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader,desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress_bar:
        
        images = batch[0].to(device)
        batch_size = images.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0,1000,(batch_size,)).to(device)
        
        # 添加噪声到图像
        noisy_images, noise = ddpm.add_noise(images,t)
        
        # 预测噪声
        predicted_noise = model(noisy_images,t)
        
        # 计算损失
        loss = loss_fn(predicted_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # 计算平均损失
    avg_loss = epoch_loss / len(train_loader.dataset)
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
# 生成样本
model.eval()

n_samples = 4
with torch.inference_mode():
    samples = ddpm.sample(model,n_samples=4,img_size=32,device=device)

samples = (samples.clamp(-1,1) + 1) / 2 # Normalize to [0, 1]
samples = samples.cpu().permute(0,2,3,1).numpy() # Change to (N,C,H,W) format

grid_size = int(np.sqrt(n_samples))
fig,axes = plt.subplots(grid_size,grid_size,figsize=(10,10))

for i,ax in enumerate(axes.flatten()):
    if i < n_samples:
        if samples.shape[-1] == 1:
            ax.imshow(samples[i].squeeze(),cmap='gray')
        else:
            ax.imshow(samples[i])
    ax.axis('off')
plt.tight_layout()
plt.show()