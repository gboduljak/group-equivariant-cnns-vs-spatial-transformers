import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.patches as patches
import torch.nn as nn


def plot_group_feature_maps(
    model: nn.Module,
    layer: int,
    target_image: int,
    target_channel: int,
    num_rotations: int = 4,
    palette="viridis",
    figsize=(4, 4)
):
  with torch.no_grad():

    test_dataset = torchvision.datasets.MNIST(
        root="./datasets",
        train=False,
        transform=None
    )

    digit, _ = test_dataset[target_image]
    digit = T.ToTensor()(digit)

    angles = torch.linspace(0, 360 - 360/num_rotations, num_rotations)
    rotated_digits_batch = torch.stack([
        TF.rotate(digit, angle.item(), TF.InterpolationMode.BILINEAR)
        for angle in angles
    ])
    rotated_digits_batch = T.Normalize(
        (0.1307,), (0.3081,))(rotated_digits_batch)

    _, intermediate_outs = model(rotated_digits_batch, True)

    target_layer_out = intermediate_outs[layer]

    fig, ax = plt.subplots(
        target_layer_out.shape[2] + 1,
        num_rotations,
        figsize=figsize
    )

    for i, angle in enumerate(angles):
      ax[0, i].imshow(
          rotated_digits_batch[i].squeeze().detach().numpy(),
          cmap="gray"
      )
      ax[0, i].set_yticks([])
      ax[0, i].set_xticks([])

    for angle_ix, angle in enumerate(angles):
      for group_ix in range(target_layer_out.shape[2]):
        ax[group_ix + 1, angle_ix].imshow(
            target_layer_out[
                angle_ix,
                target_channel,
                group_ix,
                :,
                :
            ].detach().numpy(),
            cmap=palette
        )
        ax[group_ix + 1, angle_ix].set_yticks([])
        ax[group_ix + 1, angle_ix].set_xticks([])

    for i, angle in enumerate(angles):
      patch = patches.Rectangle(
          (-0.5, 0),
          target_layer_out.shape[-1],
          target_layer_out.shape[-2],
          linewidth=4,
          edgecolor='r',
          facecolor='none'
      )
      ax[i + 1, i].add_patch(patch)
      ax[0, i].set_title(f" {int(angle.item())}°")
    fig.text(0.1, 0.825,
             "input",
             va="center",
             rotation="vertical"
             )
    fig.text(0.1, 0.415,
             "group dimension",
             va="center",
             rotation="vertical"
             )
    fig.text(0.5, 0.06,
             f"spatial dimension",
             va="center",
             ha="center"
             )
    fig.text(0.5, 0.015,
             f"feature maps (layer={layer}, channel={target_channel})",
             va="center",
             ha="center"
             )
    fig.subplots_adjust(wspace=0)
    plt.show()


def plot_feature_maps(
    model: nn.Module,
    layer: int,
    target_image: int,
    target_channel: int,
    num_rotations: int = 4,
    palette="viridis",
    figsize=(4, 4)
):
  with torch.no_grad():

    test_dataset = torchvision.datasets.MNIST(
        root="./datasets",
        train=False,
        transform=None
    )

    digit, _ = test_dataset[target_image]
    digit = T.ToTensor()(digit)

    angles = torch.linspace(0, 360 - 360/num_rotations, num_rotations)
    rotated_digits_batch = torch.stack([
        TF.rotate(digit, angle.item(), TF.InterpolationMode.BILINEAR)
        for angle in angles
    ])
    rotated_digits_batch = T.Normalize(
        (0.1307,), (0.3081,))(rotated_digits_batch)

    _, intermediate_outs = model(rotated_digits_batch, True)

    target_layer_out = intermediate_outs[layer]
    fig, ax = plt.subplots(2, num_rotations, figsize=(4, 2))

    for i, angle in enumerate(angles):
      ax[0, i].imshow(
          rotated_digits_batch[i].squeeze().detach().numpy(),
          cmap="gray"
      )
      ax[0, i].set_title(f" {int(angle.item())}°")
      ax[0, i].set_yticks([])
      ax[0, i].set_xticks([])

    for i, _ in enumerate(angles):
      ax[1, i].imshow(
          target_layer_out[i, target_channel, :, :].detach().numpy(),
          cmap=palette
      )
      ax[1, i].set_yticks([])
      ax[1, i].set_xticks([])

    fig.text(0.07, 0.65,
             "input",
             rotation="vertical"
             )

    fig.text(0.5, 0.025,
             f"feature maps (layer={layer}, channel={target_channel})",
             va="center",
             ha="center"
             )
    plt.show()
