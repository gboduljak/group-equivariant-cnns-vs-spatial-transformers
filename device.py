
import torch

mps_enabled = False


def get_device():
  return (
      torch.device("cuda") if torch.cuda.is_available()
      else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
  )


def get_accelerator():
  if torch.cuda.is_available():
    return "gpu"
  elif torch.backends.mps.is_available() and mps_enabled:
    return "mps"
  else:
    return "cpu"
