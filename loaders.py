import torchvision
from torch.utils.data.dataloader import DataLoader

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
  rotate_train = experiment_config["rotate_train"]
  rotate_test = experiment_config["rotate_test"]
  batch_size = experiment_config["batch_size"]
  train_transform = rotation_and_normalization_transform if rotate_train else normalization_transform
  test_transform = rotation_and_normalization_transform if rotate_test else normalization_transform
  train_dataset = torchvision.datasets.MNIST(
      root=datasets_root_path,
      train=True,
      transform=train_transform,
      download=False
  )
  test_dataset = torchvision.datasets.MNIST(
      root=datasets_root_path,
      train=False,
      transform=test_transform,
      download=False
  )
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            shuffle=True)
  test_loader = DataLoader(test_dataset,
                           batch_size=2048,
                           pin_memory=True,
                           shuffle=False)

  return train_loader, test_loader
