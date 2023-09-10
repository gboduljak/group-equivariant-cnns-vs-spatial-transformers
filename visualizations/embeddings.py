import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from sklearn.manifold import TSNE


def visualise_dimreduced_embeddings(model, loader, palette="viridis"):

  with torch.no_grad():
    model.eval()

    image_embeddings = []
    labels = []

    for (x, y) in iter(loader):
      image_embeddings.append(model.embed(x))
      labels.append(y)

    image_embeddings = torch.cat(image_embeddings).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    projected_image_embeddings = TSNE(
        n_components=2,
        random_state=420
    ).fit_transform(image_embeddings)
    plot_df = pd.DataFrame({
        "label": labels,
        "PC1":  projected_image_embeddings[:, 0],
        "PC2":  projected_image_embeddings[:, 1],
    })
    plot_df.reset_index(inplace=True)
    plt.rcParams.update({"font.size": 12})
    plt.figure(figsize=(8, 8))
    sns.jointplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="label",
        height=10,
        palette=sns.color_palette(palette, 10),
        legend="full"
    )


def plot_embeddings_of_rotated_image(model, target_image: int, num_rotations: int, palette="viridis", title="Image Embeddings", figsize=(8, 8)):
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
  rotated_digits_batch = T.Normalize((0.1307,), (0.3081,))(rotated_digits_batch)
  embeddings_per_digit = model.embed(rotated_digits_batch)

  fig, ax = plt.subplots(
      nrows=num_rotations,
      ncols=2,
      width_ratios=[0.1, 0.9],
      figsize=figsize
  )

  for i, angle in enumerate(angles):
    ax[i][0].imshow(
        rotated_digits_batch[i].squeeze().detach().numpy(), cmap="gray")
    ax[i][0].set_yticks([])
    ax[i][0].set_xticks([])
    ax[i][0].set_title(f" {int(angle.item())}°")
    ax[i][1].imshow(
        embeddings_per_digit[i, None, :].detach().numpy(),
        aspect="auto",
        cmap=palette
    )
    ax[i][1]
    ax[i][1].set_yticks([])

    fig.text(
        0.55,
        0.978,
        title,
        ha="center",
        va="center"
    )
    plt.tight_layout()
