import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.global_pooling import GlobalAvgPooling3d, GlobalMaxPooling3d
from models.layers.group_conv import GroupConv
from models.layers.lifting_conv import LiftingConv
from models.layers.group_spatial_max_pool import GroupSpatialMaxPool


class GroupEquivariantCNN(torch.nn.Module):

  def __init__(self, group, in_channels, out_channels, kernel_size, num_hidden, hidden_channels, global_pooling_mode="mean"):
    super().__init__()

    poolings = {
        "mean": GlobalAvgPooling3d(),
        "max": GlobalMaxPooling3d()
    }

    self.lifting_conv = LiftingConv(
        group=group,
        in_channels=in_channels,
        out_channels=hidden_channels,
        kernel_size=kernel_size
    )

    self.gconvs = torch.nn.ModuleList([
        GroupConv(
            group=group,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size
        ) for _ in range(num_hidden)
    ])

    self.poolings = torch.nn.ModuleList([
        GroupSpatialMaxPool(
            kernel_size=2,
            stride=1,
            padding=0
        ) for _ in range(num_hidden - 1)
    ])

    self.norm = lambda x: F.layer_norm(x, x.shape[-4:])

    self.global_pooling = poolings[global_pooling_mode]
    self.classifier = torch.nn.Linear(hidden_channels, out_channels)

  def embed(self, x, return_intermediate_results=False):

    x = self.lifting_conv(x)
    x = self.norm(x)
    x = F.relu(x)

    for (conv, pool) in zip(self.gconvs, self.poolings):
      x = conv(x)
      x = self.norm(x)
      x = F.relu(x)
      x = pool(x)

    x = self.gconvs[-1](x)
    x = self.norm(x)
    x = F.relu(x)

    x = self.global_pooling(x).squeeze()
    return x

  def forward(self, x):
    x = self.embed(x)
    x = self.classifier(x)
    return x
