import torchvision
import torch

from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader


class DatasetSubset(Dataset):
  def __init__(self, subset, transform=None):
    self.subset = subset
    self.transform = transform

  def __getitem__(self, index):
    x, y = self.subset[index]
    if self.transform:
      x = self.transform(x)
    return x, y

  def __len__(self):
    return len(self.subset)


normalization_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
])

rotation_and_normalization_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomRotation(
        [0, 360],
        torchvision.transforms.InterpolationMode.BILINEAR,
        fill=0),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
])


def get_loaders(datasets_root_path: str, experiment_config: dict):
  seed = experiment_config["seed"]
  rotate_train = experiment_config["rotate_train"]
  rotate_test = experiment_config["rotate_test"]
  batch_size = experiment_config["batch_size"]
  train_transform = rotation_and_normalization_transform if rotate_train else normalization_transform
  test_transform = rotation_and_normalization_transform if rotate_test else normalization_transform
  train_dataset = torchvision.datasets.MNIST(
      root=datasets_root_path,
      train=True,
      transform=None,
      download=False
  )
  test_dataset = torchvision.datasets.MNIST(
      root=datasets_root_path,
      train=False,
      transform=test_transform,
      download=False
  )
  train_subset, val_subset = random_split(
      dataset=train_dataset,
      lengths=[50000, 10000],
      generator=torch.Generator().manual_seed(seed)
  )
  train_loader = DataLoader(
      dataset=DatasetSubset(
          subset=train_subset,
          transform=train_transform
      ),
      batch_size=batch_size,
      pin_memory=True,
      shuffle=True
  )
  val_loader = DataLoader(
      dataset=DatasetSubset(
          subset=val_subset,
          transform=test_transform
      ),
      batch_size=2048,
      pin_memory=True,
      shuffle=False
  )
  test_loader = DataLoader(
      dataset=test_dataset,
      batch_size=2048,
      pin_memory=True,
      shuffle=False
  )

  return train_loader, val_loader, test_loader
