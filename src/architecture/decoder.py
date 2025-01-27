import torch.nn as nn
import torch

class Decoder(nn.Module):
    """
        Gaussian MLP Decoder
    """
    def __init__(self, latent_dim : int, hidden_dim : int, output_dim : int):
        super(Decoder, self).__init__()
        self.hidden_layer1 = nn.Linear(latent_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)


    def forward(self, z):
        z=z.to(next(self.parameters()).device)
        h = self.LeakyReLU(self.hidden_layer1(z))
        h = self.LeakyReLU(self.hidden_layer2(h))

        x_hat = torch.sigmoid(self.output_layer(h))

        return x_hat
