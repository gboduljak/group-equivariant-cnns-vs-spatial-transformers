import torch.nn as nn
import torch.nn.functional as F

from models.kernels.group_kernel import GroupKernel


class GroupConv(nn.Module):

  def __init__(self, group, in_channels, out_channels, kernel_size):
    super().__init__()

    self.kernel = GroupKernel(
        group=group,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels
    )

  def forward(self, x):
    (batch_dim, in_channels, group_dim, H_in, W_in) = x.shape
    # [batch_dim, in_channels * group_dim, H_in, W_in]
    x = x.reshape((batch_dim, in_channels * group_dim, H_in, W_in))
    # [out_channels, group_order, in_channels, group_order, kernel_size, kernel_size]
    filter_bank = self.kernel.sample_group_filter_bank()
    convolved = F.conv2d(
        input=x,
        weight=filter_bank.reshape(
            self.kernel.out_channels * self.kernel.group.order,
            self.kernel.in_channels * self.kernel.group.order,
            self.kernel.kernel_size,
            self.kernel.kernel_size
        )  # [out_channels * group_order, in_channels * group_order, group_order, kernel_size, kernel_size]
    )  # [batch_dim, out_channels * group_order, H_out, W_out]
    (_, _, H_out, W_out) = convolved.shape
    return convolved.view(
        batch_dim,
        self.kernel.out_channels,
        self.kernel.group.order,
        H_out,
        W_out
    )  # [batch_dim, out_channels, group_order, H, W]
