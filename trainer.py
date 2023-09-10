from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from device import get_accelerator


def get_trainer(experiment_config: dict):
  return Trainer(
      accelerator=get_accelerator(),
      max_epochs=experiment_config["max_epochs"],
      log_every_n_steps=experiment_config["log_every_n_steps"],
      callbacks=[
          ModelCheckpoint(
              save_weights_only=True,
              mode="max",
              monitor="val_acc",
              filename='{epoch}-{step}-{val_acc}',
          ),
          EarlyStopping(
              monitor="val_acc",
              patience=experiment_config["patience"],
              verbose=True,
              mode="max"
          )
      ],
      logger=CSVLogger(save_dir="logs/")
  )
