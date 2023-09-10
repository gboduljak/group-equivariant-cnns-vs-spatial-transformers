from torch.nn import Module


def count_model_parameters(model: Module):
  return sum(p.numel() for p in model.parameters())
