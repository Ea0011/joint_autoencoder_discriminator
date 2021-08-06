import unittest
from models.autoencoder import CosineSimilarityLoss
import torch

class CosineLossTest(unittest.TestCase):
  def test_same_class(self):
    v = torch.tensor([[1, 1], [2, 2]], dtype=torch.float32)
    labels = torch.tensor([1, 1])
    result = CosineSimilarityLoss()(v, labels).item()

    self.assertAlmostEquals(result, 0, places=4)

  def test_different_class(self):
    v = torch.tensor([[1, 1], [2, 2]], dtype=torch.float32)
    labels = torch.tensor([1, 2])
    result = CosineSimilarityLoss()(v, labels).item()

    self.assertAlmostEquals(result, 1, places=4)

  def test_same_class_wrong_direction(self):
    v = torch.tensor([[1, 1], [-1, 1]], dtype=torch.float32)
    labels = torch.tensor([1, 1])
    result = CosineSimilarityLoss()(v, labels).item()

    self.assertAlmostEquals(result, 0.5, places=4)