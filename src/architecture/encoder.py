import torch.nn as nn

class Encoder(nn.Module):
    """
        Gaussian MLP Encoder
    """
    def __init__(self, input_dim : int, hidden_dim : int, latent_dim : int):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.var_layer = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)


    def forward(self, x):
        x=x.to(next(self.parameters()).device)
        h_ = self.LeakyReLU(self.input_layer(x))
        h_ = self.LeakyReLU(self.hidden_layer(h_))
        mean = self.mean_layer(h_)
        log_var = self.var_layer(h_)

        return mean, log_var
