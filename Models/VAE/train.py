import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn,optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.mps.is_available()
                      else "cpu")

input_dim = 784
h_dim = 200
z_dim = 20
num_epochs = 20
batch_size = 128
lr = 1e-4 # Karpathy constant

# Dataset Loading
dataset = datasets.MNIST(root="../data/", download=True,transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Model, Loss and Optimizer
model = VariationalAutoEncoder(input_dim, h_dim, z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss(reduction="sum") # y_i * log(p_i) + (1-y_i) * log(1-p_i)

# Training
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader))
    for i,(x,y) in loop:
        # Forward pass
        x = x.to(device).view(x.shape[0], input_dim)
        x_reconstructed, mu, sigma = model(x)
        
        # Compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))
        
        # Backpropagation
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


model = model.to("cpu")
def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x,y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.inference_mode():
            mu, sigma = model.encode(images[d].view(1,784))
            encodings_digit.append((mu,sigma))
        
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1,1,28,28)
        save_image(out, f"generated_digit_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=3)
print("Inference complete")