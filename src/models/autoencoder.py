import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

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
    self.save_hyperparameters(hparams)
    self.input_dim = input_dim
    self.cosine_loss = CosineSimilarityLoss()

    self.prepare_decoder()
    self.prepare_encoder()
  
  def prepare_encoder(self):
    self.encoder = nn.Sequential(
      nn.Linear(self.input_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 32)
    )

    return self.encoder

  def prepare_decoder(self):
    self.decoder = nn.Sequential(
      nn.Linear(32, 256),
      nn.ReLU(),
      nn.Linear(256, 784)
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
      x_hat,
      z,
      self.hparams["cos_loss_weight"] * cosine_loss,
      self.hparams["rec_loss_weight"] * reconstruction_loss,
      self.hparams["cos_loss_weight"] * cosine_loss + self.hparams["rec_loss_weight"] * reconstruction_loss
    )

  def training_step(self, batch, batch_idx):
    x, y = batch
    (_, _, sim_loss, rec_loss, train_loss) = self.general_step(x, y)

    self.log("train_loss", train_loss)
    self.log("cosine loss", sim_loss)
    self.log("reconstruction loss", rec_loss)

    return train_loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    (x_hat, z, sim_loss, rec_loss, val_loss) = self.general_step(x, y)

    self.log("val_loss", val_loss)
    self.log("val cosine loss", sim_loss)
    self.log("val reconstruction loss", rec_loss)

    self.plot_latent(z, y)
    self.plot_reconstructions(x_hat, y)

    return val_loss

  def test_step(self, batch, batch_idx):
    pass

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
    return optimizer

  def plot_to_image(self, figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = Image.open(buf)
    # Add the batch dimension
    image = transforms.ToTensor()(image)
    return image

  def plot_latent(self, z, y):
    fig = plt.figure(figsize=(10,10))
    for i in np.unique(y):
      ix = np.where(y == i)

      plt.scatter(z[ix, 0], z[ix, 1], c=y[ix], label=i)
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])

    plt.legend()

    tensorboard = self.logger.experiment
    tensorboard.add_image("latent space", self.plot_to_image(fig), self.current_epoch)

  def plot_reconstructions(self, x, y):
    fig = plt.figure(figsize=(10,10))
    for i in range(25):
      # Start next subplot.
      plt.subplot(5, 5, i + 1, title=y[i])
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x[i].reshape(28, 28), cmap=plt.cm.binary)

    tensorboard = self.logger.experiment
    tensorboard.add_image("reconstructions", self.plot_to_image(fig), self.current_epoch)
