import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


def convert_image_np(img):
  """Taken from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html"""
  img = img.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  img = std * img + mean
  img = np.clip(img, 0, 1)
  return img


def visualize_transform_grid(model, loader, num_batches=1, grid_images=56):
  loader_iter = iter(loader)

  for _ in range(num_batches):
    with torch.no_grad():
      x, _ = next(loader_iter)
      x = x[:grid_images]
      transformed_imgut_tensor = model.stconv.stconv.transform(x)

      in_grid = convert_image_np(make_grid(x))
      out_grid = convert_image_np(make_grid(transformed_imgut_tensor))

      _, (left, right) = plt.subplots(1, 2, figsize=(12, 12))
      left.imshow(in_grid)
      left.set_title(
          "Rotated Test Images")
      left.set_xticks([])
      left.set_yticks([])
      right.imshow(out_grid)
      right.set_title(
          "Outputs of the initial spatial transformer module")
      right.set_xticks([])
      right.set_yticks([])
      plt.tight_layout()


def visualise_digit_transform(model, loader, grid_images=3,  num_batches=2, label=1):

  images = []
  labels = []

  for (img, img_label) in iter(loader):
    images.append(img)
    labels.append(img_label)

  images = torch.cat(images)
  labels = torch.cat(labels)

  images_loader = DataLoader(images[labels == label], batch_size=grid_images)
  images_loader_iter = iter(images_loader)

  for _ in range(num_batches):
    images_batch = next(images_loader_iter)
    transformed_input_tensor = model.stconv.stconv.transform(images_batch)

    original_images = convert_image_np(
        make_grid(images_batch)
    )

    transformed_images = convert_image_np(
        make_grid(transformed_input_tensor)
    )

    _, (left, right) = plt.subplots(1, 2, figsize=(8, 4),)
    left.imshow(original_images)
    left.set_xticks([])
    left.set_yticks([])
    left.set_title("Rotated dataset images")

    right.imshow(transformed_images)
    right.set_title("After the initial spatial transformer module")
    right.set_xticks([])
    right.set_yticks([])
    plt.tight_layout()
