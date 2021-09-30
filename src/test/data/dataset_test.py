import unittest
from unittest.mock import patch
from dataset.mnist_dataset import MNISTDataset


@patch('dataset.mnist_dataset.MNIST')
class TestMNISTData(unittest.TestCase):
  def test_default_init(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    MNISTDataset('/some_dir').setup(None)
    mnist_mock.assert_called_with('/some_dir', train=False, download=True, transform=None)

  def test_download(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    MNISTDataset('/some_dir', download=False).setup(None)
    mnist_mock.assert_called_with('/some_dir', train=False, download=False, transform=None)

  def test_transform(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000
    some_transform = lambda: print("Hey")

    MNISTDataset('/some_dir', transform=some_transform).setup(None)
    mnist_mock.assert_called_with('/some_dir', train=False, download=True, transform=some_transform)

  def test_default_data_split(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    dataset = MNISTDataset('/some_dir')
    dataset.setup(None)

    self.assertEqual(len(dataset.train_set), 48000) 
    self.assertEqual(len(dataset.val_set), 12000)

  def test_custom_data_split(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    dataset = MNISTDataset('/some_dir', train_val_ratio=0.9)
    dataset.setup(None)

    self.assertEqual(len(dataset.train_set), 54000) 
    self.assertEqual(len(dataset.val_set), 6000)


  @patch("dataset.mnist_dataset.DataLoader")
  def test_data_loaders(self, loader_mock, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    dataset = MNISTDataset('/some_dir')
    dataset.setup(None)

    dataset.train_dataloader()
    loader_mock.assert_called_with(dataset.train_set, batch_size=64, shuffle=True)

    dataset.val_dataloader()
    loader_mock.assert_called_with(dataset.val_set, batch_size=64, shuffle=False)

    dataset.test_dataloader()
    loader_mock.assert_called_with(dataset.test_set, batch_size=64, shuffle=False)