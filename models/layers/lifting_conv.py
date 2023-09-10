import torch.nn as nn
import torch.nn.functional as F
from models.kernels.lifting_kernel import LiftingKernel


class LiftingConv(nn.Module):

  def __init__(self, group, in_channels, out_channels, kernel_size):
    super().__init__()

    self.kernel = LiftingKernel(
        group=group,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels
    )

  def forward(self, x):
    group_order = self.kernel.group.order.item()
    # [out_channel, order, in_channels, k, k]
    filter_bank = self.kernel.sample_lifting_filter_bank()
    filter_bank = filter_bank.reshape(
        self.kernel.out_channels * group_order,
        self.kernel.in_channels,
        self.kernel.kernel_size,
        self.kernel.kernel_size
    )  # [out_channel * order, in_channels, k, k]
    convolved = F.conv2d(
        input=x,
        weight=filter_bank,
        padding=1
    )  # [batch_dim, out_channel * order, out_H, out_W]
    (batch_dim, _, out_height, out_width) = convolved.shape
    # output [batch_dim, out_channels, group_order, out_H, out_W]
    return convolved.view(
        (batch_dim, self.kernel.out_channels, group_order, out_height, out_width)
    )
