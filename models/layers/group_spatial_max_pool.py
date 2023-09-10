import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupSpatialMaxPool(nn.Module):
  def __init__(self, kernel_size: int, stride: int, padding: int) -> None:
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

  def forward(self, x):
    (_, _, group_dim, _, _) = x.shape

    pooled_feature_maps = []

    for g in range(group_dim):
      pooled_feature_map = F.max_pool2d(
          input=x[:, :, g, :],
          kernel_size=(self.kernel_size, self.kernel_size),
          stride=self.stride,
          padding=self.padding
      )  # [batch_dim, in_channels, H_out, W_out]
      pooled_feature_maps.append(pooled_feature_map.unsqueeze(2))
      # [batch_dim, in_channels, 1, H_out, W_out]

    # [batch_dim, in_channels, group_dim, H_out, W_out]
    return torch.cat(pooled_feature_maps, dim=2)
