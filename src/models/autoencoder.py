import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

# TODO: Implement Everything
class CosineSimilarityLoss(nn.Module):
  def __init__(self) -> None:
    super(CosineSimilarityLoss, self).__init__()

  def forward(self, Z, class_labels):
    Z_norm = F.normalize(Z, p=2, dim=1)
    G = torch.mm(Z_norm, Z_norm.T)
    M = G.shape[0]
    
    C = torch.eq(class_labels.unsqueeze(0), class_labels.unsqueeze(0).T)
    print(C)

    # If they have the same class similarity should be close to 1, otherwise it should be close to -1
    G = torch.where(C == True, 1 - G, 1 + G)
    print(G)
    return torch.sum(G) / M**2



class AutoEncoder(pl.LightningModule):
  def __init__(self, hparams) -> None:
    super().__init__()
    self.save_hyperparameters('hparams')
  
  def prepare_encoder(self):
    pass

  def prepare_decoder(self):
    pass

  def training_step(self, batch, batch_idx):
    pass

  def validation_setp(self, batch, batch_idx):
    pass

  def test_step(self, batch, batch_idx):
    pass

  def configure_optimizers(self):
    pass
