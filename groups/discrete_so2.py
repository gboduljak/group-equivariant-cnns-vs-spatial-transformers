import torch
import numpy as np

from groups.discrete_group import DiscreteGroup


class DiscreteSO2(DiscreteGroup):
  def __init__(self, order):
    super().__init__(
        order=torch.tensor(order),
        identity=[0.]
    )

  def elements(self):
    return torch.linspace(
        start=0,
        end=2 * np.pi * (self.order - 1) / self.order,
        steps=self.order,
        device=self.identity.device
    )

  def product(self, g, h):
    return torch.remainder(g + h, 2 * np.pi)

  def inverse(self, g):
    return torch.remainder(-g, 2 * np.pi)

  def get_identity(self):
    return self.identity

  def left_action_on_R2(self, batch_g, batch_x):
    batched_reps = torch.stack([self.matrix_representation(g) for g in batch_g])
    return torch.einsum('gik, nk -> gni', batched_reps, batch_x)

  def left_action_on_itself(self, batch_g, batch_h):
    broadcasted_batch_g = batch_g.repeat(
        batch_h.shape[0], 1)  # [batch_h, batch_g]
    batch_h_to_broadcast = batch_h.unsqueeze(-1)  # [batch_h, 1]
    return self.product(broadcasted_batch_g, batch_h_to_broadcast)

  def left_action_on_r_n_signal(self, batch_g, signal):
    batch_dim = batch_g.numel()
    C, H, W = signal.shape
    batch_grep = torch.stack([
        torch.cat(
            (self.matrix_representation(g), torch.zeros(
                2, 1, device=self.identity.device)),
            dim=1
        )
        for g in batch_g
    ])
    batch_signal = signal.repeat(batch_dim, 1, 1, 1)
    sampling_grid = torch.nn.functional.affine_grid(
        batch_grep, (batch_dim, C, H, W), align_corners=True)
    return torch.nn.functional.grid_sample(
        batch_signal,
        sampling_grid,
        mode="bilinear",
        align_corners=True
    )

  def left_action_on_group_signal(self, batch_g, signal):
    def normalize(g):
      largest_elem = 2 * np.pi * (self.order - 1) / self.order
      return (2*g / largest_elem) - 1

    batch_dim = batch_g.numel()
    C, G, H, W = signal.shape

    # Apply action on the spatial part
    batch_grep = torch.stack([
        torch.cat(
            (self.matrix_representation(g), torch.zeros(
                2, 1, device=self.identity.device)),
            dim=1
        )
        for g in batch_g
    ])  # [batch_dim, 2, 3]

    sampling_grid_x_y = torch.nn.functional.affine_grid(
        batch_grep,
        (batch_dim, C, H, W),
        align_corners=True
    )  # [batch_dim, H, W, 2]

    # Apply action on the group dimension
    sampling_grid_z = self.left_action_on_itself(
        self.inverse(batch_g),
        self.elements()
    )  # [batch_dim, G]

    # Convert to "pixel_value" in [-1, 1]
    sampling_grid_z = normalize(sampling_grid_z)  # [batch_dim, G]

    # Assemble 3D sampling grid
    sampling_grid_x_y = sampling_grid_x_y.reshape(
        (batch_dim, 1, H, W, 2))        # [batch_dim, 1, H, W, 2]
    sampling_grid_x_y = sampling_grid_x_y.repeat(
        1, self.order, 1, 1, 1)          # [batch_dim, G, H, W, 2]
    sampling_grid_z = sampling_grid_z.reshape(
        (batch_dim, self.order, 1, 1, 1))   # [batch_dim, G, 1, 1, 1]
    sampling_grid_z = sampling_grid_z.repeat(
        1, 1, H, W, 1)                       # [batch_dim, G, H, W, 1]
    sampling_grid_x_y_z = torch.cat(
        (sampling_grid_x_y, sampling_grid_z), dim=-1)  # [batch_dim, G, H, W, 3]

    # Sample the signal from translated domain given by sampling_grid_x_y_z
    # [batch_dim, C, G, H, W]
    batch_signal = signal.repeat(batch_dim, 1, 1, 1, 1)

    return torch.nn.functional.grid_sample(
        batch_signal,
        sampling_grid_x_y_z,
        mode="bilinear",
        align_corners=True
    )  # [batch_dim, C, G, H, W]

  def matrix_representation(self, g):
    cos = torch.cos(g)
    sin = torch.sin(g)

    return torch.tensor([
        [cos, -sin],
        [sin, cos]
    ], device=self.identity.device)
