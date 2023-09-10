from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from models.group_equivariant_cnn import GroupEquivariantCNN


class MNISTModule(LightningModule):

  def __init__(self,
               model_name: str,
               model_hparams: dict,
               optimizer_hparams: dict
               ):
    super().__init__()
    assert (model_name in ["GCNN", "STCNN"])
    models = {
        "GCNN": GroupEquivariantCNN
    }
    self.save_hyperparameters(
        "model_name",
        "model_hparams",
        "optimizer_hparams"
    )
    self.model = models[model_name](**model_hparams)
    self.loss = CrossEntropyLoss()

  def forward(self, imgs):
    return self.model(imgs)

  def configure_optimizers(self):
    return AdamW(self.parameters(), **self.hparams.optimizer_hparams)

  def training_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    discretized_preds = preds.argmax(dim=-1)
    loss = self.loss(preds, labels)
    accuracy = (discretized_preds == labels).float().mean()
    self.log("train_loss", loss.item(), prog_bar=True)
    self.log("train_acc", accuracy.item(), prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    discretized_preds = preds.argmax(dim=-1)
    loss = self.loss(preds, labels)
    accuracy = (labels == discretized_preds).float().mean()
    self.log("val_loss", loss.item(), prog_bar=True)
    self.log("val_acc", accuracy.item(), prog_bar=True)

  def test_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    discretized_preds = preds.argmax(dim=-1)
    loss = self.loss(preds, labels)
    accuracy = (labels == discretized_preds).float().mean()
    self.log("test_loss", loss.item(), prog_bar=True)
    self.log("test_acc", accuracy.item(), prog_bar=True)
