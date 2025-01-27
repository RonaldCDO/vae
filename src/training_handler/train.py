from architecture.encoder import Encoder
from architecture.decoder import Decoder
from architecture.model import Vae

from dotenv import load_dotenv
import os

from torch.optim import Adam
import torch.nn as nn
import torch

load_dotenv()

class Trainer:
    def loss_function(self, x, x_hat, mean, log_var, beta=1.0):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + beta * KLD

    def train(self, epochs: int, train_loader, val_loader, model, optimizer):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x, _ in train_loader:
                x = x.view(x.size(0), -1).to(DEVICE)
                optimizer.zero_grad()

                x_hat, mean, log_var = model.forward(x)

                loss = self.loss_function(x, x_hat, mean, log_var)

                train_loss += loss.item()

                loss.backward()

                optimizer.step()
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x , _ in val_loader:
                    x = x.view(x.size(0), -1).to(DEVICE)
                    x_hat, mean, log_var = model(x)
                    loss = self.loss_function(x, x_hat, mean, log_var)
                    val_loss += loss.item()
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"{os.getenv('WEIGHTS_PATH')}/final_weights.pth")
        print("Model weights saved to weights/vae_weights.pth")
        return train_losses, val_losses


                
