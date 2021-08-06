import torchvision
from torchvision.datasets import MNIST
import torch

class MNISTDataset():
  def __init__(self, root_dir, train_val_ratio=0.8, download=True, transform=None) -> None:
      mnist_data_train = MNIST(root_dir, train=True, download=download, transform=transform)
      self.test_set = MNIST(root_dir, train=False, download=download, transform=transform)

      train_size = round(len(mnist_data_train) * train_val_ratio)
      train_set, val_set = torch.utils.data.random_split(mnist_data_train, [train_size, len(mnist_data_train) - train_size])

      self.train_set = train_set
      self.val_set = val_set
