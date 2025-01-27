from training_handler.train import Trainer
from training_handler.inference import visualize_reconstruction
from architecture.model import Vae
from architecture.decoder import Decoder
from architecture.encoder import Encoder
from dataset.loader import DatasetLoader
from graph_utils.plot import plot_training_graph

import os
import sys
from dotenv import load_dotenv

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch


load_dotenv()

DEVICE = os.getenv('DEVICE', 'cuda')
DATASET_PATH = os.getenv('DATASET_PATH')
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH')

def main():

    input_dim = 784
    hidden_dim = 400
    latent_dim = 200
    epochs = 30 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = DatasetLoader(root=DATASET_PATH, transform=transform, train=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = DatasetLoader(root=DATASET_PATH, transform=transform, train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


    trainer = Trainer()

    encoder = Encoder(
                    input_dim=input_dim, 
                    hidden_dim=hidden_dim, 
                    latent_dim=latent_dim,
                ).to(DEVICE)
    decoder = Decoder(
                    latent_dim=latent_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=input_dim,
                ).to(DEVICE)

    model = Vae(encoder, decoder)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'inference':
            model.load_state_dict(torch.load(f'{WEIGHTS_PATH}/final_weights.pth', weights_only=True))
            visualize_reconstruction(model, val_loader)
        elif sys.argv[1] == 'train':
            train_losses, val_losses = trainer.train(
                epochs=epochs, 
                train_loader=train_loader, 
                val_loader=val_loader,
                model=model, 
                optimizer=optimizer,
            )

            plot_training_graph(train_losses, val_losses, epochs)

if __name__ == "__main__":
    main()
