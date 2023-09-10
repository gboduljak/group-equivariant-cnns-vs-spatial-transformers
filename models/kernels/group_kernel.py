import torch
import torch.nn as nn
import math


class GroupKernel(nn.Module):
  def __init__(self, group, kernel_size, in_channels, out_channels):
    super().__init__()

    self.group = group
    self.kernel_size = kernel_size
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.weight = nn.Parameter(torch.zeros((
        self.out_channels,
        self.in_channels,
        self.group.order,
        self.kernel_size,
        self.kernel_size
    ), device=self.group.device))

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))

  def sample_group_filter_bank(self):
    filter_bank = self.group.left_action_on_group_signal(
        batch_g=self.group.elements(),
        signal=self.weight.view((
            self.out_channels * self.in_channels,
            self.group.order,
            self.kernel_size,
            self.kernel_size
        ))  # [out_channels * in_channels, group_order, kernel_size, kernel_size]
    )  # [group_order, out_channels * in_channels, group_order, kernel_size, kernel_size]

    filter_bank = filter_bank.view((
        self.group.order,
        self.out_channels,
        self.in_channels,
        self.group.order,
        self.kernel_size,
        self.kernel_size
    ))  # [group_order, out_channels, in_channels, group_order, kernel_size, kernel_size]

    # [out_channels, group_order, in_channels, group_order, kernel_size, kernel_size]
    return filter_bank.permute(1, 0, 2, 3, 4, 5)
