import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

load_dotenv()

def plot_training_graph(train_losses, val_losses, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label = 'Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label = 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{os.getenv('GRAPHS_PATH')}/training_validation.png')
