import torch.nn as nn
import torch.nn.functional as F
from models.localization_net import LocalizationNet


class STConv(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               localization_channels,
               localization_initialization_mode="identity",
               transformation_mode="affine"
               ):
    super().__init__()
    self.conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size
    )
    self.localization_net = LocalizationNet(
        in_channels=in_channels,
        kernel_size=kernel_size,
        localization_channels=localization_channels,
        initialization_mode=localization_initialization_mode,
        transformation_mode=transformation_mode
    )

  def localize(self, x):
    return self.localization_net(x)

  def transform(self, x):
    theta = self.localize(x)
    sampling_grid = F.affine_grid(theta, size=x.shape)
    return F.grid_sample(input=x, grid=sampling_grid)

  def forward(self, x):
    return self.conv(self.transform(x))
