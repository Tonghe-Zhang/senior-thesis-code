import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Initialize the model
dim = 16  # Example dimension
model = SinusoidalPosEmb(dim)

# Create a range of input values
x = torch.arange(0, 100, dtype=torch.float32)  # Example input range

# Get the sinusoidal embeddings
embeddings = model(x)

# Plot the embeddings
plt.figure(figsize=(12, 8))
for i in range(dim):
    plt.plot(x.cpu().numpy(), embeddings[:, i].cpu().numpy(), label=f'Dim {i}')
plt.xlabel('Input Value')
plt.ylabel('Embedding Value')
plt.title('Sinusoidal Positional Embeddings')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
