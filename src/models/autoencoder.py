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

    # If they have the same class similarity should be close to 1, otherwise it should be close to -1
    G = torch.where(C == True, 1 - G, 1 + G)
    return torch.sum(G) / M**2



class AutoEncoder(pl.LightningModule):
  def __init__(self, hparams, input_dim) -> None:
    super().__init__()
    self.save_hyperparameters('hparams')
    self.input_dim = input_dim
    self.cosine_loss = CosineSimilarityLoss()
  
  def prepare_encoder(self):
    self.encoder = nn.Sequential(
      nn.Linear(self.input_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 16)
    )

    return self.encoder

  def prepare_decoder(self):
    self.decoder = nn.Sequential(
      nn.Linear(16, 256),
      nn.ReLU(),
      nn.Linear(256, 1024)
    )

    return self.decoder

  def forward(self, x):
    z = self.encoder(x)
    return z

  def general_step(self, x, y):
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    cosine_loss = self.cosine_loss(z, y)

    x_hat = self.decoder(z)
    reconstruction_loss = F.l1_loss(x, x_hat)

    return (
      self.hparams["cos_loss_weight"] * cosine_loss,
      self.hparams["rec_loss_weight"] * reconstruction_loss,
      self.hparams["cos_loss_weight"] * cosine_loss + self.hparams["rec_loss_weight"] * reconstruction_loss
    )

  def training_step(self, batch, batch_idx):
    x, y = batch
    (_, _, train_loss) = self.general_step(x, y)

    self.log("train_loss", train_loss)

    return train_loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    (_, _, val_loss) = self.general_step(x, y)

    self.log("val_loss", val_loss)

    return val_loss

  def test_step(self, batch, batch_idx):
    pass

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
    return optimizer
