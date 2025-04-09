import torch
import torch.optim as optim
import torch.nn.functional as F
from model import ConditionalVAE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='../data/',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='../data/',
    train=False,
    transform=transform,
    download=True
)

# 创建数据加载器
batch_size = 128
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.mps.is_available()
                      else "cpu")

# 初始化模型
input_dim = 784  # 28x28 = 784
condition_dim = 10  # 10个数字类别
cvae = ConditionalVAE(input_dim=input_dim, condition_dim=condition_dim).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=1e-3)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1)  # 展平图像
        data = data.to(device)
        
        # 将标签转换为one-hot编码
        targets_onehot = F.one_hot(targets, num_classes=condition_dim).float().to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, log_var = cvae(data, targets_onehot)
        loss, recon_loss, kl_loss = cvae.loss_function(data, recon_batch, mu, log_var)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    
# 生成一些样本（每个数字生成一个样本）
with torch.no_grad():
    for digit in range(10):
        # 创建one-hot编码
        c = F.one_hot(torch.tensor([digit]), num_classes=condition_dim).float().to(device)
        sample = cvae.sample(1, c).cpu().view(28, 28)
            
        plt.subplot(2, 5, digit+1)
        plt.imshow(sample.numpy(), cmap='gray')
        plt.title(f'Digit {digit}')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f'outputs.png')
    plt.close()