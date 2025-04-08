import torch
import torchvision.datasets as datasets
from torch import nn,optim
from model import ConditionalVAE
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.mps.is_available()
                      else "cpu")

input_dim = 784 # 28*28
condition_dim = 10 # 对于one-hot编码的数字标签
h_dim = 200
z_dim = 20
num_epochs = 20
batch_size = 128
lr = 1e-4 # Karpathy constant

# Dataset Loading
dataset = datasets.MNIST(root="../data/", download=True,transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss and Optimizer
model = ConditionalVAE(input_dim,condition_dim, h_dim, z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(model,dataloader,optimizer,epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx,(data,labels) in enumerate(dataloader):
            # 将图像展平为784维向量
            data_flattened = data.view(-1,input_dim).to(device)
            
            # 将标签转换为one-hot编码
            condition = torch.zeros(data.shape[0],condition_dim)
            condition.scatter_(1, labels.unsqueeze(1),1)
            condition = condition.to(device)
            
            # 前向传播  
            reconstruction, mu, log_var = model(data_flattened,condition)
            
            # 计算损失
            loss = model.loss_function(reconstruction, data_flattened, mu, log_var)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 限制梯度范数
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader.dataset):.4f}")

def generate_samples(model,digits=range(10),n_samples_per_digit=1):
    model.eval()
    samples = []
    
    with torch.inference_mode():
        for digit in digits:
            # 为每个数字创建条件向量
            condition = torch.zeros(n_samples_per_digit,condition_dim)
            condition[:,digit] = 1
            condition = condition.to(device)
            
            # 从模型中采样
            digit_samples = model.sample(n_samples_per_digit,condition)
            samples.append(digit_samples)
    
    return torch.cat(samples,dim=0)

print(f"Training on {device}...")
train(model, train_loader, optimizer, num_epochs)

# 生成样本
plt.figure(figsize=(15, 3))
samples = generate_samples(model, num_samples_per_digit=1)
samples = samples.cpu().view(-1, 28, 28).numpy()

for i, img in enumerate(samples):
    plt.subplot(1, 10, i+1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Digit: {i}")

plt.tight_layout()
plt.savefig("vae_samples.png")
plt.show()