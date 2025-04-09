import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from model import UNet, DDPM

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 将图像大小调整为32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型
num_classes = 10  # MNIST有10个类别
model = UNet(
    input_channels=1,  # MNIST是灰度图像
    hidden_dims=64,
    time_emb_dim=128,
    num_classes=num_classes  # 启用条件生成
).to(device)
ddpm = DDPM(n_steps=1000,beta_start=1e-4,beta_end=0.02).to(device)

# 设置训练参数
learning_rate = 2e-4
num_epochs = 30

# 初始化优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for batch_idx, (images, labels) in progress_bar:
        images = images.to(device) # 形状为[batch_size, 1, 32, 32]
        labels = labels.to(device) # 形状为[batch_size]
        
        batch_size = images.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, ddpm.n_steps, (batch_size,), device=device)
        
        # 执行训练步骤
        loss = ddpm.train_step(model, optimizer, images, t, labels)
        
        # 更新进度条
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")
        epoch_loss += loss
    
    # 计算平均损失
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] 完成, 平均损失: {avg_loss:.6f}")
    
    # 更新学习率
    scheduler.step()

print("训练完成！")

model.eval()

# 创建更大的可视化图表
rows, cols = 4, 10
fig, axes = plt.subplots(rows, cols, figsize=(20, 8))

for row in range(rows):
    for col in range(cols):
        digit = col  # 每列代表一个数字
        with torch.no_grad():
            condition = torch.tensor([digit], device=device)
            
            # 为每个数字生成多个不同的样本
            sample = ddpm.sample(model, 1, 32, device, condition)
            
            img = sample[0].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            
            axes[row, col].imshow(img.squeeze(), cmap='gray')
            if row == 0:
                axes[row, col].set_title(f"{digit}")
            axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("final_samples.png")
plt.show()