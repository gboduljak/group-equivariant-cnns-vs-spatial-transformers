
import torch


class DiscreteGroup(torch.nn.Module):
  def __init__(self, order, identity, device):
    super().__init__()
    self.order = order
    self.register_buffer("identity", torch.Tensor(identity, device=device))
    self.device = device
