import os
from dotenv import load_dotenv
load_dotenv()

import torch

import matplotlib.pyplot as plt

DEVICE=os.getenv('DEVICE', 'cuda') 

def visualize_reconstruction(model, data_loader):
    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.view(x.size(0), -1).to(DEVICE)  # Flatten input
            reconstruction, _, _ = model(x)
            reconstruction = reconstruction.view(-1, 1, 28, 28)  # Reshape for visualization
            x = x.view(-1, 1, 28, 28)
            break

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        axes[0, i].imshow(x[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    print('Saving inference reconstruction to graphs/visualize_reconstruction.png')
    plt.savefig(f'{os.getenv('GRAPHS_PATH')}/visualize_reconstruction.png')

