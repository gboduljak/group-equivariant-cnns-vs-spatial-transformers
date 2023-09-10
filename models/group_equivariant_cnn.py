import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.global_pooling import GlobalAvgPooling3d, GlobalMaxPooling3d
from models.layers.group_conv import GroupConv
from models.layers.lifting_conv import LiftingConv
from models.layers.group_spatial_max_pool import GroupSpatialMaxPool
from groups.discrete_group import DiscreteGroup


class LiftingConvBlock(nn.Module):
  def __init__(self,
               group: DiscreteGroup,
               in_channels: int,
               out_channels: int,
               kernel_size: int) -> None:
    super().__init__()

    self.lift = LiftingConv(
        group=group,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size
    )
    self.norm = lambda x: F.layer_norm(x, x.shape[-4:])
    self.relu = lambda x: F.relu(x)

  def forward(self, x):
    x = self.lift(x)
    x = self.norm(x)
    x = self.relu(x)
    return x


class GroupConvBlock(nn.Module):
  def __init__(self,
               group: DiscreteGroup,
               in_channels: int,
               out_channels: int,
               kernel_size: int) -> None:
    super().__init__()

    self.gconv = GroupConv(
        group=group,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size
    )
    self.norm = lambda x: F.layer_norm(x, x.shape[-4:])
    self.relu = lambda x: F.relu(x)

  def forward(self, x):
    x = self.gconv(x)
    x = self.norm(x)
    x = self.relu(x)
    return x


class GroupEquivariantCNN(nn.Module):

  def __init__(self,
               group,
               in_channels,
               out_channels,
               kernel_size,
               gconv_layers,
               channels,
               global_pooling_mode="mean"
               ):
    super().__init__()

    poolings = {
        "mean": GlobalAvgPooling3d(),
        "max": GlobalMaxPooling3d()
    }

    self.lift = LiftingConvBlock(
        group=group,
        in_channels=in_channels,
        out_channels=channels,
        kernel_size=kernel_size
    )

    self.gconvs = torch.nn.ModuleList([
        GroupConvBlock(
            group=group,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size
        ) for _ in range(gconv_layers)
    ])

    self.poolings = torch.nn.ModuleList([
        GroupSpatialMaxPool(
            kernel_size=2,
            stride=1,
            padding=0
        ) for _ in range(gconv_layers - 1)
    ])

    self.global_pooling = poolings[global_pooling_mode]
    self.classifier = torch.nn.Linear(channels, out_channels)

  def embed(self, x, return_intermediate_results=False):
    intermediate_results = []

    def save(x):
      if return_intermediate_results:
        intermediate_results.append(x)

    x = self.lift(x)
    save(x)
    for (conv, pool) in zip(self.gconvs, self.poolings):
      x = conv(x)
      save(x)
      x = pool(x)
      save(x)
    x = self.gconvs[-1](x)
    save(x)
    x = self.global_pooling(x).squeeze()
    save(x)
    if return_intermediate_results:
      return x, intermediate_results
    else:
      return x

  def forward(self, x, return_intermediate_results=False):
    if not return_intermediate_results:
      x = self.embed(x)
      x = self.classifier(x)
      return x
    else:
      x, results = self.embed(x, return_intermediate_results)
      x = self.classifier(x)
      return x, results + [x]
