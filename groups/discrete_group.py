
import torch


class DiscreteGroup(torch.nn.Module):
  def __init__(self, order, identity):
    super().__init__()
    self.order = order
    self.register_buffer("identity", torch.tensor(identity))
