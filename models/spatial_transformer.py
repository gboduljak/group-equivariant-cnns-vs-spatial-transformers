import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.global_pooling import GlobalAvgPooling2d, GlobalMaxPooling2d
from models.layers.st_conv import STConv


class ConvBlock(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int):
    super().__init__()
    self.conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size
    )
    self.norm = lambda x: F.layer_norm(x, x.shape[-3:])
    self.relu = lambda x: F.relu(x)

  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    x = self.relu(x)
    return x


class STConvBlock(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               localization_channels: int,
               kernel_size: int,
               localization_initialization_mode="identity",
               transformation_mode="rotation"):
    super().__init__()

    self.stconv = STConv(
        in_channels=in_channels,
        out_channels=out_channels,
        localization_channels=localization_channels,
        localization_initialization_mode=localization_initialization_mode,
        transformation_mode=transformation_mode,
        kernel_size=kernel_size
    )
    self.norm = lambda x: F.layer_norm(x, x.shape[-3:])
    self.relu = lambda x: F.relu(x)

  def forward(self, x):
    x = self.stconv(x)
    x = self.norm(x)
    x = self.relu(x)
    return x


class STCNN(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               conv_layers,
               channels,
               localization_channels,
               mode="single",
               localization_initialization_mode="identity",
               transformation_mode="rotation",
               global_pooling_mode="mean"
               ):
    super().__init__()

    poolings = {
        "mean": GlobalAvgPooling2d(),
        "max": GlobalMaxPooling2d()
    }

    self.stconv = STConvBlock(
        in_channels=in_channels,
        out_channels=channels,
        localization_channels=localization_channels,
        kernel_size=kernel_size,
        localization_initialization_mode=localization_initialization_mode,
        transformation_mode=transformation_mode
    )
    if mode == "single":
      self.convs = torch.nn.ModuleList([
          ConvBlock(
              in_channels=channels,
              out_channels=channels,
              kernel_size=kernel_size
          ) for _ in range(conv_layers - 1)
      ])
    else:
      self.convs = torch.nn.ModuleList([
          STConvBlock(
              in_channels=channels,
              out_channels=channels,
              localization_channels=localization_channels,
              kernel_size=kernel_size,
              localization_initialization_mode=localization_initialization_mode,
              transformation_mode=transformation_mode
          ) for _ in range(conv_layers - 1)
      ])

    self.poolings = torch.nn.ModuleList([
        nn.MaxPool2d(
            kernel_size=2,
            stride=1,
            padding=0
        ) for _ in range(conv_layers - 2)
    ])

    self.global_pooling = poolings[global_pooling_mode]
    self.classifier = torch.nn.Linear(channels, out_channels)

  def embed(self, x, return_intermediate_results=False):
    intermediate_results = []

    def save(x):
      if return_intermediate_results:
        intermediate_results.append(x)

    x = self.stconv(x)
    save(x)
    for (conv, pool) in zip(self.convs, self.poolings):
      x = conv(x)
      save(x)
      x = pool(x)
      save(x)
    x = self.convs[-1](x)
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
