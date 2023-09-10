import torch
import math
import torch.nn as nn
from groups.discrete_group import DiscreteGroup


class LiftingKernel(nn.Module):
  def __init__(self, group: DiscreteGroup, kernel_size, in_channels, out_channels):
    super().__init__()
    self.group = group
    self.kernel_size = kernel_size
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.weight = torch.nn.Parameter(
        torch.zeros((
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        ), device=self.group.device)
    )
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))

  def sample_lifting_filter_bank(self):
    weight_signal = self.weight.view((self.out_channels * self.in_channels, self.kernel_size,
                                      self.kernel_size))  # [out_channels * in_channels, kernel_size, kernel_size]
    transformed_weight_signal = self.group.left_action_on_r_n_signal(
        batch_g=self.group.elements(),
        signal=weight_signal
    )  # [group_order, out_channels * in_channels, kernel_size, kernel_size]
    transformed_weight_signal = transformed_weight_signal.view((
        self.group.order,
        self.out_channels,
        self.in_channels,
        self.kernel_size,
        self.kernel_size
    ))  # [group_order, out_channels, in_channels, kernel_size, kernel_size]
    # [out_channels, group_order, in_channels, kernel_size, kernel_size]
    filter_bank = transformed_weight_signal.permute(1, 0, 2, 3, 4)
    # [group_order, out_channels, in_channels, kernel_size, kernel_size]
    return filter_bank
