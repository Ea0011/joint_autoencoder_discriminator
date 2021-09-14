import torchvision
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class MNISTDataset(pl.LightningDataModule):
  def __init__(self, root_dir, batch_size=64, train_val_ratio=0.8, download=True, transform=None) -> None:
    super().__init__()
    self.batch_size = batch_size
    self.train_val_ratio = train_val_ratio
    self.download = download
    self.transform = transform
    self.root_dir = root_dir

  def setup(self, stage):
    mnist_data_train = MNIST(self.root_dir, train=True, download=self.download, transform=self.transform)
    self.test_set = MNIST(self.root_dir, train=False, download=self.download, transform=self.transform)

    train_size = round(len(mnist_data_train) * self.train_val_ratio)
    train_set, val_set = torch.utils.data.random_split(mnist_data_train, [train_size, len(mnist_data_train) - train_size])

    self.train_set = train_set
    self.val_set = val_set

  def train_dataloader(self):
    return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

