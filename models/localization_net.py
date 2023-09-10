import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.layers.global_pooling import GlobalMaxPooling2d, GlobalAvgPooling2d


class LocalizationNet(nn.Module):
  def __init__(self,
               in_channels,
               kernel_size,
               localization_channels,
               initialization_mode="random",
               transformation_mode="affine",
               global_pooling="mean"
               ):
    super().__init__()
    pooling = {"max": GlobalMaxPooling2d(), "mean": GlobalAvgPooling2d()}
    assert (transformation_mode in ["affine", "rotation"])
    assert (initialization_mode in ["random", "identity"])
    self.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=localization_channels,
        kernel_size=kernel_size
    )
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, stride=2)
    self.conv2 = nn.Conv2d(
        in_channels=localization_channels,
        out_channels=localization_channels,
        kernel_size=kernel_size)
    self.relu2 = nn.ReLU()
    self.global_pool = pooling[global_pooling]

    if transformation_mode == "rotation":
      self.register_buffer("cos_mask", torch.tensor(
          [[1, 0, 0], [0, 1, 0]]).float())
      self.register_buffer("sin_mask", torch.tensor(
          [[0, -1, 0], [1, 0, 0]]).float())
      self.regressor = nn.Sequential(
          nn.Linear(in_features=localization_channels, out_features=1),
          nn.Sigmoid()
      )  # regressing just angle for rotation
    else:
      self.regressor = nn.Sequential(
          nn.Linear(
              in_features=localization_channels,
              out_features=6
          )
      )
    if initialization_mode == "identity":
      regressor_linear, *_ = self.regressor
      regressor_linear.weight.data.zero_()
      if transformation_mode == "affine":
        regressor_linear.bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0],
                         dtype=torch.float))
      else:
        regressor_linear.bias.data.copy_(
            torch.tensor([pow(10, 5)],
                         dtype=torch.float))
    self.transformation_mode = transformation_mode

  def forward(self, x):
    batch_dim = x.size(0)
    x = self.conv1(x)
    x = F.layer_norm(x, x.shape[-3:])
    x = self.relu1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = F.layer_norm(x, x.shape[-3:])
    x = self.relu2(x)
    x = self.global_pool(x)
    x = self.regressor(x)

    if self.transformation_mode == "affine":
      return x.view((batch_dim, 2, 3))
    else:
      angle = x.view(batch_dim, ) * 2 * math.pi
      cos = torch.cos(angle)
      sin = torch.sin(angle)
      cos_part = torch.einsum('ij,n->nij', self.cos_mask, cos)
      sin_part = torch.einsum('ij,n->nij', self.sin_mask, sin)
      return cos_part + sin_part
