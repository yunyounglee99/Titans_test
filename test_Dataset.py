import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class RandomTokenDataset(Dataset):
  def __init__(self, num_samples, seq_length, vocab_size):
    self.num_samples = num_samples
    self.seq_length = seq_length
    self.vocab_size = vocab_size

  def __len__(self):
    return self.num_samples
  
  def __getitem__(self, idx):
    return torch.randint(0, self.vocab_size, (self.seq_length,))
  
  