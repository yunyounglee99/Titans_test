import torch
from titans_pytorch import NeuralMemory

device = torch.device("mps")

mem = NeuralMemory(
  dim = 384,
  chunk_size = 64
).to(device)

seq = torch.randn(2, 1024, 384).to(device)
retrieved, mem_state = mem(seq)

assert seq.shape == retrieved.shape