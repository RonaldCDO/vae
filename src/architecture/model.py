from architecture.decoder import Decoder
from architecture.encoder import Encoder
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import os
from dotenv import load_dotenv

load_dotenv()


class Vae(nn.Module):
    """
        Implementation of the Variational AutoEncoder model
    """
    def __init__(self, encoder : Encoder, decoder : Decoder):
        super(Vae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = self.encoder.input_dim

    def reparameterize(self, mean : float, var: float):
        DEVICE=os.getenv('DEVICE', 'cuda') 
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mean, log_var =self.encoder.forward(x)
        z = self.reparameterize(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder.forward(z)

        return x_hat, mean, log_var
