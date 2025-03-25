import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer

# Define a Transformer model from scratch
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, hidden_dim, n_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.transformer = Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.encoder(src)
        output = self.transformer(src, src)
        return self.decoder(output)

# Initialize Model
model = TransformerModel(input_dim=5000, output_dim=10, n_heads=8, hidden_dim=512, n_layers=6)

# Dummy Input
src = torch.randint(0, 5000, (10, 32))  # Sequence length 10, batch size 32
output = model(src)
print(output.shape)  # Output shape
