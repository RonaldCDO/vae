from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DatasetLoader(MNIST):
    def __init__(
                self, 
                root : str, 
                train : bool = True, 
                transform = None, 
                download : bool = True,
                ):

        super().__init__(
            root=root, 
            train=train, 
            transform=transform, 
            download=download,
        )
