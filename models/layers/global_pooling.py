import torch
import torch.nn as nn


class GlobalAvgPooling2d(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.mean(x, dim=(-2, -1))


class GlobalAvgPooling3d(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.mean(x, dim=(-3, -2, -1))


class GlobalMaxPooling2d(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.amax(x, dim=(-2, -1))


class GlobalMaxPooling3d(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.amax(x, dim=(-3, -2, -1))
