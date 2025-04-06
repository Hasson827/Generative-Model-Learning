import torch
from torch import nn

# Input img -> Hidden dim -> mean,std -> Parametrization trick -> Reparameterized trick -> Decoder -> Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        
        # Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim,z_dim)
        self.hid_2sigma = nn.Linear(h_dim,z_dim)
        
        # Decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU()

        
    def encode(self, x):
        """
        Encode the input x into a latent representation.
        q_phi(z|x)
        """
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    
    def decode(self, z):
        """
        Decode the latent representation z back to the original space.
        p_theta(x|z)
        """
        h = self.relu(self.z_2hid(z))
        x_reconstructed = torch.sigmoid(self.hid_2img(h))
        return x_reconstructed
    
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
    

if __name__=="__main__":
    x = torch.randn(4,28*28)
    vae = VariationalAutoEncoder(input_dim=28*28)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)