import torch

device = torch.device("mps")

is_mps_available = torch.mps.is_available()
print(f"MPS available: {is_mps_available}")

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)

print(f"PyTorch {torch.__version__} on {device}")
print(a @ b)
