import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.patches as patches
import torch.nn as nn


def plot_equivariant_kernels(
    model: nn.Module,
    layer: int,
    target_image: int,
    out_channel: int,
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

    out_layer = intermediate_outs[layer]

    fig, ax = plt.subplots(
        out_layer.shape[2],
        angles.numel(),
        figsize=figsize
    )

    # for i, angle in enumerate(angles):
    #   ax[0, i].imshow(
    #       rotated_digits_batch[i].squeeze().detach().numpy(),
    #       cmap="gray"
    #   )
    #   ax[0, i].set_yticks([])
    #   ax[0, i].set_xticks([])

    for angle_ix, angle in enumerate(angles):
      for group_ix in range(out_layer.shape[2]):
        ax[group_ix, angle_ix].imshow(
            out_layer[angle_ix, out_channel,
                      group_ix, :, :].detach().numpy(),

        )
        ax[group_ix, angle_ix].set_yticks([])
        ax[group_ix, angle_ix].set_xticks([])

    for i, angle in enumerate(angles):
      patch = patches.Rectangle(
          (-0.5, 0),
          out_layer.shape[-1],
          out_layer.shape[-2],
          linewidth=4,
          edgecolor='r',
          facecolor='none'
      )
      ax[i, i].add_patch(patch)
      ax[0, i].set_title(f" {int(angle.item())}°")

    fig.text(0.07, 0.5,
             "group dimension",
             va="center",
             rotation="vertical"
             )
    fig.text(0.5, 0.05,
             "feature maps",
             va="center",
             ha="center"
             )
    fig.subplots_adjust(wspace=0)
    plt.show()
