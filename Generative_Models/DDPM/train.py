import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from model import UNet,DDPM

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.mps.is_available()
                      else "cpu")

transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = UNet(input_channels=1,hidden_dims=64).to(device)
ddpm = DDPM(n_steps=1000).to(device)

lr = 3e-4
num_epochs = 10

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        images = images.to(device)
        condition = torch.zeros(images.shape[0],10)
        condition.scatter_(1, labels.unsqueeze(1), 1)
        condition = condition.to(torch.long).to(device)
        batch_size = images.shape[0]

        # 随机采样时间步
        t = torch.randint(0,1000,(batch_size,)).to(device)
        
        # 添加噪声到图像
        noisy_images, noise = ddpm.add_noise(images,t)
        
        # 预测噪声
        predicted_noise = model(noisy_images,t,condition)
        
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

plt.figure(figsize=(15,16))
for digit in range(10):
    plt.subplot(2,5,digit+1)
    with torch.inference_mode():
        
        condition = torch.zeros(1,10)
        condition[:,digit] = 1
        condition = condition.to(torch.long).to(device)
        
        samples = ddpm.sample(model,1,32,device,condition)
        
    img = samples[0].cpu().permute(1,2,0).numpy()
    img = (img.squeeze()+1)/2 # 反归一化到[0,1]
    
    plt.imshow(img,cmap='gray')
    plt.title(f"Digit: {digit}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("generated_samples.png")
plt.show()