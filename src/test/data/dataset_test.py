import unittest
from unittest.mock import patch
from dataset.mnist_dataset import MNISTDataset


@patch('dataset.mnist_dataset.MNIST')
class TestMNISTData(unittest.TestCase):
  def test_default_init(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    MNISTDataset('/some_dir')
    mnist_mock.assert_called_with('/some_dir', train=False, download=True, transform=None)

  def test_download(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    MNISTDataset('/some_dir', download=False)
    mnist_mock.assert_called_with('/some_dir', train=False, download=False, transform=None)

  def test_transform(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000
    some_transform = lambda: print("Hey")

    MNISTDataset('/some_dir', transform=some_transform)
    mnist_mock.assert_called_with('/some_dir', train=False, download=True, transform=some_transform)

  def test_default_data_split(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    dataset = MNISTDataset('/some_dir')
    self.assertEqual(len(dataset.train_set), 48000) 
    self.assertEqual(len(dataset.val_set), 12000)

  def test_custom_data_split(self, mnist_mock):
    mnist_mock.return_value = [None] * 60000

    dataset = MNISTDataset('/some_dir', train_val_ratio=0.9)
    self.assertEqual(len(dataset.train_set), 54000) 
    self.assertEqual(len(dataset.val_set), 6000)